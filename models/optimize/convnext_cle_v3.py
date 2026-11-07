"""ConvNeXt-CLE V3 —— R3 轮候选创新点 (2026-11-08)

设计文档: experiments/R3_DESIGN.md。在 V2(WR+CH, convnext_v2bc) 基础上继续叠加/推广,
全部保持 V2 的两条铁律: 结构化替换(计算图不可旁路) + 严格加载审计(预训练零损失)。

候选(工厂名即消融链):
  v3f  FWR 全深度小波替换: WTConv 替换推广到 stage-1..4 全部 18 个 block
       (stage-1/2 同样经 _remap_wr_all 继承预训练 dwconv; V1 时代"s234 更差"的
        结论来自可旁路的零门控并联, 不可信, 本候选在替换语义下重测)。
  v3s  SOP 二阶纹理池化头: CH 一阶 GAP 只保留通道均值, 丢弃通道共现(纹理=特征共现,
       腺体排列规则性 = 通道响应的联合结构)。对 s2/s3/s4 各做 1x1 投影(d=64) →
       去均值 → 协方差 → Newton-Schulz 逆平方根幂归一化(iSQRT-COV 风格, 参数 free,
       fp32 计算) → 展平, 与一阶 CH 拼接向量共同送入分类器(头结构改变=不可旁路)。
  v3x  XRM 自上而下语义门控精炼: 池化前用 s4 全局语义(预训练最终 LN 输出)对 s2/s3
       特征做乘性通道门控 f'=f⊙(1+tanh(W·g4)), 纯乘法无加性恒等旁路(区别于 V1 被
       证伪的零门控并联注入: 门控作用于被消费路径本身)。
  v3r  DRC 腺体尺度扩张卷积 @stage-2(28²层): stage-2 的 7x7 dwconv 替换为
       base(k7, 继承预训练) + 双膨胀分支(k3 d=3/d=6, 尺度初始 0.1), 为高分辨率层
       提供腺体簇尺度的上下文(结构同 WR 的 base+扩展范式)。

新增参数段白名单: sop.* / xrm.* / drc_*; 详见 train_cle_fair.py 注册。
"""
import torch
import torch.nn as nn

from models.Convnext import ConvNeXt, Block
from models.optimize.convnext_cle import WTConv2d, _ScaleModule
from models.optimize.convnext_cle_v2 import WaveletReplaceBlock


# ═══════════════ 候选 2: SOP 二阶纹理池化 ═══════════════

class SecondOrderPool(nn.Module):
    """iSQRT-COV 风格二阶池化: 多尺度特征 → 幂归一化协方差展平。

    每尺度: 1x1 投影到 d 维 → token 去均值 → C = X̃X̃ᵀ/N (B,d,d, fp32) →
    Frobenius 归一 + Newton-Schulz 逆平方根(5 次迭代, 参数 free) → 恢复幅值 →
    展平 (B, d²)。协方差捕获通道共现(纹理组织), 幂归一化抑制 burstiness。
    """

    def __init__(self, dims, d=64, iters=5):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(c, d, 1, bias=False) for c in dims])
        self.d, self.iters = d, iters

    def forward(self, feats):
        outs = []
        eye = torch.eye(self.d, device=feats[0].device)
        for proj, f in zip(self.projs, feats):
            x = proj(f).flatten(2).float()                    # (B,d,N)
            x = x - x.mean(-1, keepdim=True)
            n = x.shape[-1]
            cov = torch.bmm(x, x.transpose(1, 2)) / n         # (B,d,d) PSD
            norm = cov.flatten(1).norm(dim=-1).view(-1, 1, 1) + 1e-8
            cov = cov / norm
            cov = cov + 1e-2 * eye                            # 秩亏正则(加大: NS 梯度放大 ~λ^-3/2)
            s = eye.unsqueeze(0).expand(cov.shape[0], -1, -1).contiguous()
            for _ in range(self.iters):                       # S ← (3S − S³C)/2 → C^(-1/2)
                s = 1.5 * s - 0.5 * torch.bmm(torch.bmm(torch.bmm(s, s), s), cov)
            out = (s * norm.detach().pow(0.5)).flatten(1)     # 幅值回缩 detach(梯度稳)
            outs.append(torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0))
        return torch.cat(outs, dim=-1)                        # (B, 3d²)


# ═══════════════ 候选 3: XRM 自上而下乘性门控 ═══════════════

class TopDownGate(nn.Module):
    """g4(预训练最终 LN 输出) → 各浅层乘性通道门: f' = f ⊙ (1+tanh(W·g4))。"""

    def __init__(self, d_sem, dims):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(d_sem, c) for c in dims])
        for lin in self.linears:
            nn.init.normal_(lin.weight, std=0.02)             # 初始门 ≈ 1(轻微扰动)
            nn.init.zeros_(lin.bias)

    def forward(self, g4, feats):
        return [f * (1 + torch.tanh(lin(g4)).view(f.shape[0], -1, 1, 1))
                for lin, f in zip(self.linears, feats)]


# ═══════════════ 候选 4: DRC 多尺度膨胀替换块 ═══════════════

class MultiDilConv(nn.Module):
    """base(k7, 预训练继承) + 双膨胀 k3 分支(d=3 覆盖 ~19px / d=6 覆盖 ~37px 原生)。"""

    def __init__(self, dim):
        super().__init__()
        self.base = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=True)
        self.drc_d3 = nn.Conv2d(dim, dim, 3, padding=3, dilation=3, groups=dim, bias=False)
        self.drc_d6 = nn.Conv2d(dim, dim, 3, padding=6, dilation=6, groups=dim, bias=False)
        self.drc_s3 = _ScaleModule([1, dim, 1, 1], init_scale=0.1)
        self.drc_s6 = _ScaleModule([1, dim, 1, 1], init_scale=0.1)

    def forward(self, x):
        return self.base(x) + self.drc_s3(self.drc_d3(x)) + self.drc_s6(self.drc_d6(x))


class DilatedReplaceBlock(Block):
    """stage-2 block: 仅替换 dwconv 为 MultiDilConv, 其余与原 Block 一致。"""

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__(dim, drop_rate, layer_scale_init_value)
        self.dwconv = MultiDilConv(dim)


# ═══════════════ 组合网络 ═══════════════

class ConvNeXtV3(ConvNeXt):
    """ConvNeXt + {WR@stages, DRC@stages, XRM, CH/SOP 头} 可组合。"""

    def __init__(self, wr_stages=(), drc_stages=(), use_xrm=False,
                 head="ch", sop_dim=64, **kw):
        super().__init__(**kw)
        dims0 = self.downsample_layers[0][0].out_channels
        self.dims = [dims0] + [self.downsample_layers[i][1].out_channels for i in (1, 2, 3)]

        for si in wr_stages:
            self.stages[si] = nn.Sequential(
                *[WaveletReplaceBlock(self.dims[si]) for _ in self.stages[si]])
        for si in drc_stages:
            self.stages[si] = nn.Sequential(
                *[DilatedReplaceBlock(self.dims[si]) for _ in self.stages[si]])

        self.use_xrm = use_xrm
        if use_xrm:
            self.xrm = TopDownGate(self.dims[3], [self.dims[1], self.dims[2]])

        self.head_mode = head
        d4 = self.dims[3]
        if head == "ch":
            self.msf2_p2 = nn.Conv2d(self.dims[1], d4, 1, bias=False)
            self.msf2_p3 = nn.Conv2d(self.dims[2], d4, 1, bias=False)
            self.msf2_n2 = nn.LayerNorm(d4, eps=1e-6)
            self.msf2_n3 = nn.LayerNorm(d4, eps=1e-6)
            self.head = nn.Linear(3 * d4, self.head.out_features)
        elif head == "sop":
            self.msf2_p2 = nn.Conv2d(self.dims[1], d4, 1, bias=False)
            self.msf2_p3 = nn.Conv2d(self.dims[2], d4, 1, bias=False)
            self.msf2_n2 = nn.LayerNorm(d4, eps=1e-6)
            self.msf2_n3 = nn.LayerNorm(d4, eps=1e-6)
            self.sop = SecondOrderPool([self.dims[1], self.dims[2], d4], d=sop_dim)
            self.sop_ln = nn.LayerNorm(3 * sop_dim * sop_dim, eps=1e-6)
            self.head = nn.Linear(3 * d4 + 3 * sop_dim * sop_dim, self.head.out_features)
        else:
            raise ValueError(head)

    def forward_features(self, x):
        feats = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feats.append(x)
        m4 = self.norm(feats[3].mean([-2, -1]))               # 预训练最终 LN 保留
        if self.use_xrm:
            f2, f3 = self.xrm(m4.detach(), feats[1:3])        # 语义向量仅作门控条件
        else:
            f2, f3 = feats[1], feats[2]
        m3 = self.msf2_n3(self.msf2_p3(f3).mean([-2, -1]))
        m2 = self.msf2_n2(self.msf2_p2(f2).mean([-2, -1]))
        v1 = torch.cat([m4, m3, m2], dim=-1)                  # 一阶 CH 通路
        if self.head_mode == "sop":
            v2 = self.sop_ln(self.sop([f2, f3, feats[3]]).to(v1.dtype))
            return torch.cat([v1, v2], dim=-1)
        return v1


_TINY = dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])


def _v3(num_classes, **kw):
    return ConvNeXtV3(num_classes=num_classes, **{**_TINY, **kw})


def convnext_v3f(num_classes: int):   # FWR: 全深度小波替换 + CH
    return _v3(num_classes, wr_stages=(0, 1, 2, 3), head="ch")


def convnext_v3s(num_classes: int):   # WR@s34 + SOP 二阶池化头(含一阶通路)
    return _v3(num_classes, wr_stages=(2, 3), head="sop")


def convnext_v3x(num_classes: int):   # WR@s34 + XRM 语义门控 + CH
    return _v3(num_classes, wr_stages=(2, 3), use_xrm=True, head="ch")


def convnext_v3r(num_classes: int):   # WR@s34 + DRC@stage2 + CH
    return _v3(num_classes, wr_stages=(2, 3), drc_stages=(1,), head="ch")


# ═══════════════ v2c 底座变体 (CA 批结论: WR 拖累 CH, 以 CH 为底座叠加) ═══════════════

def convnext_v3cs(num_classes: int):  # v2c + SOP 二阶池化头
    return _v3(num_classes, wr_stages=(), head="sop")


def convnext_v3cx(num_classes: int):  # v2c + XRM 语义门控
    return _v3(num_classes, wr_stages=(), use_xrm=True, head="ch")


def convnext_v3cr(num_classes: int):  # v2c + DRC@stage2
    return _v3(num_classes, wr_stages=(), drc_stages=(1,), head="ch")
