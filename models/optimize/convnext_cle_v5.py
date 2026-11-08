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
        s_att = torch.sigmoid((le - le.mean([2, 3], keepdim=True))
                              / (le.var([2, 3], keepdim=True) + self.eps))
        dev = (high - high.mean([2, 3], keepdim=True)).pow(2)
        e_att = torch.sigmoid((dev - dev.mean([2, 3], keepdim=True))
                              / (dev.var([2, 3], keepdim=True) + self.eps))
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
        g = torch.sigmoid((dev - dev.mean([2, 3], keepdim=True)) / (2 * std + 1e-5))
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
                 use_xrm=False, pre_module=None, pre_stages=(),
                 use_fgs=False, use_csap=False, **kw):
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
        if use_fgs:                                          # FGS: 茎干梯度增广(零init等价)
            self.downsample_layers[0][0] = FGStem(self.downsample_layers[0][0])
        self.use_csap = use_csap
        self.use_xrm = use_xrm
        if use_xrm:
            self.xrm = TopDownGate2(self.dims[3], [self.dims[1], self.dims[2]])
        d2, d3, d4 = self.dims[1], self.dims[2], self.dims[3]
        if use_csap:                                         # CSAP: 替换均匀池化(不建 msf2)
            self.csap2 = CSAHead(d2)
            self.csap3 = CSAHead(d3)
            self.head = nn.Linear(3 * d4, self.head.out_features)
            return
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
        if self.use_csap:
            m3 = self.csap3(f3, m4)
            m2 = self.csap2(f2, m4)
            return torch.cat([m4, m3, m2], dim=-1)
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


class CSAHead(nn.Module):
    """CSAP 创新点: 跨尺度语义条件化注意力池化。

    对 s2/s3 每个空间位置打分: score = MLP([P_s·f_s(x,y) ; g4]), softmax(带温度)为
    空间注意力 α, 池化特征 = Σ α·(P_s·f_s)。s4 保持预训练 LN 的 GAP(语义自证)。
    动机: 暗背景 25-31% 稀释均匀 GAP; 多灶异型证据集中(局部异质性 5.2-5.7x);
    临床读片"多灶肿瘤取最差病灶"。完全替换均匀池化 → 不可旁路。
    """

    def __init__(self, d_s, d_sem=768):
        super().__init__()
        self.proj = nn.Conv2d(d_s, d_sem, 1, bias=False)
        self.score = nn.Sequential(nn.Linear(2 * d_sem, d_sem), nn.GELU(), nn.Linear(d_sem, 1))
        self.logit_scale = nn.Parameter(torch.zeros(1))      # softmax 温度 s=exp(0)=1
        self.norm = nn.LayerNorm(d_sem, eps=1e-6)

    def forward(self, f, g4):                                # f:(N,C,H,W) g4:(N,d_sem)
        z = self.proj(f)                                     # (N,d,H,W)
        b, d, h, w = z.shape
        zf = z.flatten(2).transpose(1, 2)                    # (N,HW,d)
        g = g4.unsqueeze(1).expand(-1, h * w, -1)            # (N,HW,d)
        s = self.score(torch.cat([zf, g], -1)).squeeze(-1)   # (N,HW)
        a = torch.softmax(s * self.logit_scale.exp(), dim=-1)
        return self.norm((zf * a.unsqueeze(-1)).sum(1))


class FGStem(nn.Module):
    """FGS 创新点: 前景/梯度增广茎干 —— 输入 [R,G,B,Sobel 幅值], 梯度通道零初始化
    (初始与预训练严格逐位等价; CLE 数据 RGB 同源, r=g=b 时原响应不变)。"""

    def __init__(self, conv):
        super().__init__()
        self.conv = conv                                     # 原 Conv2d(3,C,4,4), 权重膨胀到 4ch
        with torch.no_grad():
            w = self.conv.weight
            new_w = torch.zeros(w.shape[0], 4, *w.shape[2:], device=w.device, dtype=w.dtype)
            new_w[:, :3] = w
            self.conv.weight = nn.Parameter(new_w)
        kx = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        self.register_buffer("sobel_x", kx[None, None], persistent=False)
        self.register_buffer("sobel_y", kx.t()[None, None].contiguous(), persistent=False)

    def forward(self, x):                                    # (N,3,H,W)
        g = x.float().mean(1, keepdim=True)
        gx = torch.nn.functional.conv2d(g, self.sobel_x, padding=1)
        gy = torch.nn.functional.conv2d(g, self.sobel_y, padding=1)
        mag = torch.hypot(gx, gy)
        mag = (mag - mag.mean(dim=(2, 3), keepdim=True)) / (mag.std(dim=(2, 3), keepdim=True) + 1e-5)
        return self.conv(torch.cat([x, mag.to(x.dtype)], 1))


def convnext_v5xrcw_factory(num_classes: int):  # 占位, 实际经 extra --class-weights
    return _v5(num_classes, use_xrm=True)


def convnext_v6c(num_classes: int):    # 创新2: CSAP 语义条件注意力池化头(替换均匀GAP)
    return _v5(num_classes, use_csap=True)


def convnext_v6f(num_classes: int):    # 创新2/3: FGS 梯度增广茎干(零init等价预训练)
    return _v5(num_classes, use_fgs=True)


def convnext_v6cf(num_classes: int):   # CSAP + FGS 组合
    return _v5(num_classes, use_csap=True, use_fgs=True)


def convnext_v6f2(num_classes: int):  # FGS(独立别名, 测试目录隔离用)
    return _v5(num_classes, use_fgs=True)
