"""ConvNeXt-CLE V5 —— R4 轮: 在 v2c(CH) 底座上叠加第二创新点候选 (2026-11-08)

用户指令: 放弃 CSM(降点), 以 v2c=CH(96.38/κ94.60, 3s) 为底座找净涨 ≥0.1-0.2pt 的组件,
重点参考 module_library, 也可考虑损失函数。全部替换式/预训练严格加载审计。

候选(工厂名即消融链, 均含 CH):
  v5se   SE 通道门控 @stage-4(库 43 SE 家族): 残差分支 MLP 输出后乘性通道注意力
         (GAP→FC(C/16)→ReLU→FC→sigmoid, y=x⊙s)。多乘法路径不可旁路; +0.22M。
  v5grn  GRN 全局响应归一化 @stage-3/4(ConvNeXtV2 官方设计): pwconv1 与 GELU 之间
         插入跨 token 的通道竞争归一化(γ·x/‖x‖₂+β, 2C 参数/块)。
  v5telu TeLU 激活替换 @stage-3/4(库 111, arXiv 2025): GELU→x·tanh(exp(x)) (零参数,
         exp 输入 clamp 至 10 防 fp16 溢出; x>10 时渐近恒等)。
  v5xrm2 XRM 温和重试: 自上而下乘性语义门控(v3cx 的 s42 曾单点 0.9686 全场最高但
         s43 崩, 判定为门控初始扰动过大 std0.02→门幅±0.5; 本版 std 0.005→门幅±0.12)。
损失函数探针(focal 需管线支持, ls/cw 用现有参数): 见 jobs_CI.json 的 extra 字段。
"""
import torch
import torch.nn as nn

from models.Convnext import ConvNeXt, Block


def telu(x):
    return x * torch.tanh(torch.exp(x.float().clamp(max=10.))).to(x.dtype)


class SEGate(nn.Module):
    """SE 通道门控(库 43 SE 家族), 作用于 (…, C) 特征。"""

    def __init__(self, dim, r=16):
        super().__init__()
        self.se_fc1 = nn.Linear(dim, dim // r)
        self.se_fc2 = nn.Linear(dim // r, dim)
        nn.init.normal_(self.se_fc2.weight, std=0.01)
        nn.init.constant_(self.se_fc2.bias, 2.0)            # s≈0.88 起步, 温和衰减

    def forward(self, x):                                   # (N,H,W,C)
        s = x.mean(dim=(1, 2))
        s = torch.sigmoid(self.se_fc2(torch.relu(self.se_fc1(s))))
        return x * s.unsqueeze(1).unsqueeze(2)


class GRN(nn.Module):
    """ConvNeXtV2 全局响应归一化: 通道间竞争(跨 token 的 L2 范数归一 + 仿射)。"""

    def __init__(self, dim):
        super().__init__()
        self.grn_gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.grn_beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):                                   # (N,H,W,C)
        gx = torch.norm(x.float(), p=2, dim=(1, 2), keepdim=True)
        nx = x / (gx.to(x.dtype) + 1e-6)
        return self.grn_gamma * nx + self.grn_beta + x      # 残差式: γ=β=0 时恒等保持预训练


class TopDownGate2(nn.Module):
    """v5xrm2: 温和初始化(std 0.005)的自上而下乘性通道门控。"""

    def __init__(self, d_sem, dims, std=0.005):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(d_sem, c) for c in dims])
        for lin in self.linears:
            nn.init.normal_(lin.weight, std=std)
            nn.init.zeros_(lin.bias)

    def forward(self, g4, feats):
        return [f * (1 + torch.tanh(lin(g4)).view(f.shape[0], -1, 1, 1))
                for lin, f in zip(self.linears, feats)]


class PFESA(nn.Module):
    """库 223 (MICCAI 2026) 忠实移植: FFT 高斯掩码分离低频结构/高频边缘,
    双分支无参统计注意力相加再 sigmoid, 乘性重标定。fp32 内部计算。"""

    def __init__(self, base_ratio=0.1):
        super().__init__()
        self.base_ratio = base_ratio
        self.eps = 1e-5

    def forward(self, x):                                   # (N,C,H,W)
        h, w = x.shape[-2:]
        xf = x.float()
        f = torch.fft.fftshift(torch.fft.fftn(xf, dim=(-2, -1)), dim=(-2, -1))
        r = self.base_ratio * min(h, w) / max(h, w)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                                torch.linspace(-1, 1, w, device=x.device), indexing="ij")
        mask = torch.exp(-(yy ** 2 + xx ** 2) / (2 * r ** 2))
        low = torch.abs(torch.fft.ifftn(torch.fft.ifftshift(f * mask, dim=(-2, -1)), dim=(-2, -1)))
        high = torch.abs(torch.fft.ifftn(torch.fft.ifftshift(f * (1 - mask), dim=(-2, -1)), dim=(-2, -1)))
        le = low ** 2
        s_att = torch.sigmoid((le - le.mean([2, 3], True)) / (le.var([2, 3], True) + self.eps))
        dev = (high - high.mean([2, 3], True)).pow(2)
        e_att = torch.sigmoid((dev - dev.mean([2, 3], True)) / (dev.var([2, 3], True) + self.eps))
        return (torch.sigmoid(s_att + e_att) * xf).to(x.dtype)


class SDGW(nn.Module):
    """库 233 (CVPR 2026) 精神移植(无参温和版): 局部窗口统计 -> 偏差引导乘性门控,
    调制范围限制在 (0.5, 1.5) 保持预训练特征量级。"""

    def forward(self, x):                                   # (N,C,H,W)
        xf = x.float()
        mu = torch.nn.functional.avg_pool2d(xf, 3, 1, 1)
        m2 = torch.nn.functional.avg_pool2d(xf * xf, 3, 1, 1)
        std = (m2 - mu * mu + 1e-5).sqrt()
        dev = (xf - mu).abs()
        g = torch.sigmoid((dev - dev.mean([2, 3], True)) / (2 * std + 1e-5))
        return (xf * (0.5 + g)).to(x.dtype)


class V5Block(Block):
    """ConvNeXt Block + 可选 {SE 门控 / GRN / TeLU / dwconv 后无参重标定}; dwconv/LN/残差不变。"""

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6,
                 use_se=False, use_grn=False, use_telu=False, pre_module=None):
        super().__init__(dim, drop_rate, layer_scale_init_value)
        self.use_se, self.use_grn, self.use_telu = use_se, use_grn, use_telu
        self.pre_module = pre_module
        if use_se:
            self.se = SEGate(dim)
        if use_grn:
            self.grn = GRN(4 * dim)                          # 作用于 pwconv1 输出(4C 隐宽)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        if self.pre_module is not None:
            x = self.pre_module(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        if self.use_grn:
            x = self.grn(x)
        x = telu(x) if self.use_telu else self.act(x)
        x = self.pwconv2(x)
        if self.use_se:
            x = self.se(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXtV5(ConvNeXt):
    """ConvNeXt-T + CH + {SE/GRN/TeLU@stages, XRM} 可组合。"""

    def __init__(self, mod_stages=(), use_se=False, use_grn=False, use_telu=False,
                 use_xrm=False, pre_module=None, pre_stages=(), **kw):
        super().__init__(**kw)
        dims0 = self.downsample_layers[0][0].out_channels
        self.dims = [dims0] + [self.downsample_layers[i][1].out_channels for i in (1, 2, 3)]
        for si in mod_stages:
            self.stages[si] = nn.Sequential(
                *[V5Block(self.dims[si], use_se=use_se, use_grn=use_grn, use_telu=use_telu)
                  for _ in self.stages[si]])
        if pre_module is not None:                           # dwconv 后无参重标定(PFESA/SDGW)
            for si in pre_stages:
                self.stages[si] = nn.Sequential(
                    *[V5Block(self.dims[si], pre_module=pre_module())
                      for _ in self.stages[si]])
        self.use_xrm = use_xrm
        if use_xrm:
            self.xrm = TopDownGate2(self.dims[3], [self.dims[1], self.dims[2]])
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
        if self.use_xrm:
            f2, f3 = self.xrm(m4.detach(), feats[1:3])
        else:
            f2, f3 = feats[1], feats[2]
        m3 = self.msf2_n3(self.msf2_p3(f3).mean([-2, -1]))
        m2 = self.msf2_n2(self.msf2_p2(f2).mean([-2, -1]))
        return torch.cat([m4, m3, m2], dim=-1)


_TINY = dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])


def _v5(num_classes, **kw):
    return ConvNeXtV5(num_classes=num_classes, **{**_TINY, **kw})


def convnext_v5se(num_classes: int):    # CH + SE@stage-4
    return _v5(num_classes, mod_stages=(3,), use_se=True)


def convnext_v5grn(num_classes: int):   # CH + GRN@stage-3/4
    return _v5(num_classes, mod_stages=(2, 3), use_grn=True)


def convnext_v5telu(num_classes: int):  # CH + TeLU@stage-3/4
    return _v5(num_classes, mod_stages=(2, 3), use_telu=True)


def convnext_v5xrm2(num_classes: int):  # CH + XRM(温和初始化)
    return _v5(num_classes, use_xrm=True)


def convnext_v5grn4(num_classes: int):  # CH + GRN 仅@stage-4(缩小扰动范围重试)
    return _v5(num_classes, mod_stages=(3,), use_grn=True)


def convnext_v5grnx(num_classes: int):  # CH + GRN@stage-4 + XRM(两个稳定微增益组件叠加)
    return _v5(num_classes, mod_stages=(3,), use_grn=True, use_xrm=True)


def convnext_v5pf(num_classes: int):   # CH + PFESA@stage-3/4(库223 无参边缘-结构注意力)
    return _v5(num_classes, pre_module=PFESA, pre_stages=(2, 3))


def convnext_v5sd(num_classes: int):   # CH + SDGW@stage-2/3(库233 局部偏差引导加权, 无参)
    return _v5(num_classes, pre_module=SDGW, pre_stages=(1, 2))


def convnext_v5xrcw_factory(num_classes: int):  # 占位, 实际经 extra --class-weights
    return _v5(num_classes, use_xrm=True)
