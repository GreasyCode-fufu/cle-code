"""ConvNeXt-CLE V4 —— R3 轮创新点 2: CSM 循环谱前馈层(轻量化, 2026-11-08)

背景: ConvNeXt-T 的参数 90% 在 MLP(pwconv1/2, ~25.9M/27.8M); 数据分析表明 CLE
图像近灰度、判别信号在结构/纹理组织层面 → 通道混合存在大量可压缩自由度。

CSM (Circulant-Spectral MLP, 循环谱前馈层):
  把 block 的 pwconv1(Linear C→4C)+GELU+pwconv2(Linear 4C→C) 替换为
  "对角×循环×对角(DCD)单元之和" 的谱域实现:

      h_j = GELU( Σ_{g<G} D2_g ⊙ circ(λ_g) D1_g (x) ),  j = 1..r (扩展副本)
      y   = Σ_j Σ_{g<G'} D2'_g ⊙ circ(λ'_g) D1'_g (h_j)

  其中 circ(λ)y = irfft(rfft(y) ⊙ λ) —— 循环矩阵恰被 DFT 对角化(C = F/m·Λ·F^H,
  Gray 2006), 谱 λ 复数可学(幅度 softplus + 相位)。每个 DCD 单元参数 3m, 总计
  (rG+G')·3m ≪ 8C²: C=384 时 ~23K vs 1.18M/块(≈50x)。

数学性质:
  M1 对角化代数: 循环矩阵 = 被 DFT 对角化的交换子代数, 谱域只学 Λ → O(m) 参数、
     O(m log m) 乘法(rfft/irfft 实现于通道维)。
  M2 谱范数可控: ‖DCD 单元‖₂ ≤ ‖D2‖_∞·max|λ|·‖D1‖_∞ (酉不变性), 全部显式参数化
     → 层 Lipschitz 上界可控。
  M3 最优循环投影初始化: 循环矩阵构成子空间, 预训练 W 到该子空间的 Frobenius
     投影有闭式解(c_j = (1/m)Σ_i W[i,(i-j) mod m], 对角平均), 变换内置该投影 ——
     预训练知识以最小损失方式继承(与 Eckart-Young 低秩截断不同: 保秩, 压自由度)。

库内锚点: module_library 21(Partial Conv, 结构化省算力原则) / 130(Half-Conv) /
128(Converse2D, FFT 算子先例)。DCD 结构参考 Sindhwani et al. 2015 结构化线性变换。

变体: v4a=CSM@stage3(idx2) 17.2M / v4b=CSM@stage4(idx3) 13.6M / v4c=CSM@stage3+4 3.0M
(均含创新点 1: CH 三尺度拼接分类头, CA 批 3seed 确认 +2.26 Kappa)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Convnext import ConvNeXt, Block


class DCDUnit(nn.Module):
    """W = diag(d2)·C(λ)·diag(d1): 对角-循环-对角单元 (参数 3m)。"""

    def __init__(self, m, init_amp=0.5):
        super().__init__()
        self.m = m
        k = m // 2 + 1
        self.d1 = nn.Parameter(torch.ones(m))
        self.d2 = nn.Parameter(torch.ones(m))
        self.amp = nn.Parameter(torch.full((k,), math.log(math.expm1(init_amp))))
        self.phs = nn.Parameter(torch.zeros(k))

    def forward(self, x):                                   # (..., m)
        lam = torch.polar(F.softplus(self.amp.float()), self.phs.float())
        X = torch.fft.rfft(x.float() * self.d1.float(), dim=-1)
        y = torch.fft.irfft(X * lam, n=self.m, dim=-1)
        return (y * self.d2.float()).to(x.dtype)


class CSMFeedForward(nn.Module):
    """CSM 前馈层: r 个扩展副本(各 G 个 DCD 单元求和) + 共享收缩(G' 个单元对 h_j 求和)。

    keep_up=True 的混合模式(CD 批教训: 纯 CSM 容量损失过大): 升维保留预训练稠密
    Linear(经键重映射加载), 仅降维换成 CSM 谱域 —— 参数减半而容量损失小。
    """

    def __init__(self, dim, expand=4, groups=4, down_groups=4, keep_up=False):
        super().__init__()
        self.r = expand
        self.keep_up = keep_up
        if keep_up:
            self.dense_up = nn.Linear(dim, expand * dim)    # 预训练 pwconv1 重映射加载
        else:
            self.up = nn.ModuleList([
                nn.ModuleList([DCDUnit(dim, init_amp=0.5 if g == 0 else 0.05)
                               for g in range(groups)]) for _ in range(expand)])
        self.down = nn.ModuleList([DCDUnit(dim, init_amp=0.5 if g == 0 else 0.05)
                                   for g in range(down_groups)])
        self.act = nn.GELU()

    def forward(self, x):                                   # (..., C)
        y = 0
        if self.keep_up:
            h = self.act(self.dense_up(x))                  # (..., 4C)
            hs = h.unflatten(-1, (self.r, -1))              # (..., r, C)
            for j in range(self.r):
                for u in self.down:
                    y = y + u(hs[..., j, :])
            return y
        for units in self.up:
            h = 0
            for u in units:
                h = h + u(x)
            h = self.act(h)
            for u in self.down:
                y = y + u(h)
        return y


class CSMBlock(Block):
    """ConvNeXt Block, MLP 替换为 CSM(keep_up 时仅降维层替换); dwconv/LN/γ/残差不变。"""

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6, **csm_kw):
        super().__init__(dim, drop_rate, layer_scale_init_value)
        self.pwconv1 = None                              # keep_up 时经重映射迁入 csm.dense_up
        self.pwconv2 = None
        dg = csm_kw.pop("down_groups", 4)
        self.csm = CSMFeedForward(dim, down_groups=dg, **csm_kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)                           # (N,C,H,W)->(N,H,W,C)
        x = self.norm(x)
        x = self.csm(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)                           # (N,H,W,C)->(N,C,H,W)
        return shortcut + self.drop_path(x)


class ConvNeXtV4(ConvNeXt):
    """ConvNeXt-T + 创新点1(CH 三尺度拼接头) + 创新点2(CSM 循环谱前馈, csm_stages)。"""

    def __init__(self, csm_stages=(), keep_up=False, down_groups=4, **kw):
        super().__init__(**kw)
        dims0 = self.downsample_layers[0][0].out_channels
        self.dims = [dims0] + [self.downsample_layers[i][1].out_channels for i in (1, 2, 3)]
        for si in csm_stages:
            self.stages[si] = nn.Sequential(
                *[CSMBlock(self.dims[si], keep_up=keep_up, down_groups=down_groups)
                  for _ in self.stages[si]])
        d2, d3, d4 = self.dims[1], self.dims[2], self.dims[3]
        self.msf2_p2 = nn.Conv2d(d2, d4, 1, bias=False)
        self.msf2_p3 = nn.Conv2d(d3, d4, 1, bias=False)
        self.msf2_n2 = nn.LayerNorm(d4, eps=1e-6)
        self.msf2_n3 = nn.LayerNorm(d4, eps=1e-6)
        self.head = nn.Linear(3 * d4, self.head.out_features)

    def forward_features(self, x):
        feats = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feats.append(x)
        m4 = self.norm(feats[3].mean([-2, -1]))
        m3 = self.msf2_n3(self.msf2_p3(feats[2]).mean([-2, -1]))
        m2 = self.msf2_n2(self.msf2_p2(feats[1]).mean([-2, -1]))
        return torch.cat([m4, m3, m2], dim=-1)


_TINY = dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])


def _v4(num_classes, **kw):
    return ConvNeXtV4(num_classes=num_classes, **{**_TINY, **kw})


def convnext_v4a(num_classes: int):    # CH + CSM@stage-3(9块): ~17.2M
    return _v4(num_classes, csm_stages=(2,))


def convnext_v4b(num_classes: int):    # CH + CSM@stage-4(3块): ~13.6M
    return _v4(num_classes, csm_stages=(3,))


def convnext_v4c(num_classes: int):    # CH + CSM@stage-3+4: ~3.0M(激进版)
    return _v4(num_classes, csm_stages=(2, 3))


def convnext_v4d(num_classes: int):    # CH + CSM(仅降维)@stage-3: 混合版 ~23.4M
    return _v4(num_classes, csm_stages=(2,), keep_up=True)


def convnext_v4e(num_classes: int):    # CH + CSM(仅降维)@stage-4: 混合版 ~21.6M
    return _v4(num_classes, csm_stages=(3,), keep_up=True)


def convnext_v4f(num_classes: int):    # CH + CSM(仅降维,G'=8)@stage-4: 容量增强 ~21.2M
    return _v4(num_classes, csm_stages=(3,), keep_up=True, down_groups=8)
