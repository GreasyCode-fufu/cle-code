"""EPNet 候选 —— 数据驱动的三创新点(结构化, 不可旁路), 2026-11-01

设计依据(experiments/EPNET_DESIGN.md + cle_image_analysis.py):
  1. M→B 漏诊样本高频占比低 15%(0.0086 vs 0.0101), 但良性本身高频最多(B 0.0112>M 0.0103),
     => 频率处理必须"结构条件化", 不能盲目增强高频。
  2. 局部异质性 max/mean 5.15~5.70: GAP 之外的 max 证据真实存在。
  3. CLE speckle 噪声空间不均一 + 腺体方向性结构。

三创新点(全部结构化, 遵循前作"零门控可旁路注入被 wd 压死"的教训):
  A  DPH  Dual-Pooling Multi-scale Head(决策端, 前作 CH 的升级):
      每尺度 s2/s3/s4 取 [GAP; GMP] 投影拼接 -> 6*768=4608-d -> Linear。
      GAP=全局统计证据, GMP=最显著局部异型证据(多灶肿瘤"最差病灶定级")。
      head 输入维度结构改变, 强制消费双池化证据, 不可旁路。
  B  FESR Frequency Edge-Structure Recalibration(深层, stage3/4 每块 dwconv 后):
      FFT 高斯掩码分离 低频=结构S / 高频=边缘E, PFESA 式空间统计归一化,
      交互门控 g = sigmoid(w1*S + w2*E) (w init 1.0, 非零), x' = x*(1+g)。
      建模"规则结构下的边缘(良性腺缘) vs 紊乱结构下的边缘(恶性异型)"。
      参数 ~2 标量/块; 乘性重标定无零初始化, 不可旁路。
  C  LSDG Local-Statistics Direction-Gated conv(浅层 stage2 每块 dwconv 后):
      H/V 条带深度卷积(k=7)方向边缘 -> 1x1 融合 -> 局部3x3统计归一化门控
      g = sigmoid((|e|-mu)/std)(参数free, 由局部信噪比决定) -> x' = x + e*g。
      dwconv 预训练保留(不替换), 模块只在其后增强; fuse conv std=0.02 温和启动。

预训练兼容: 全部主干段(stem/downsample/dwconv/norm/pwconv)路径与形状不变,
仅新增段(dph_*/fesr_*/lsdg_*)随机初始化 + head 本就按类数新建。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Convnext import ConvNeXt, Block

_EPS = 1e-5


def _gaussian_lowpass(h: int, w: int, device, dtype, base_ratio: float = 0.1):
    """PFESA 式二维高斯低频掩码, 中心=直流, 越远越衰减。"""
    r = base_ratio * min(h, w) / max(h, w)
    y = torch.linspace(-1, 1, h, device=device, dtype=torch.float32)
    x = torch.linspace(-1, 1, w, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    m = torch.exp(-(gy ** 2 + gx ** 2) / (2 * r * r))
    return m.to(dtype)


class FESR(nn.Module):
    """B: 频域边缘-结构交互重标定(参数~2标量)。输入输出 (B,C,H,W) 同形。

    v2(2026-11-01 快筛反馈): 原 x*(1+sigmoid) 初始幅值放大~73%, 破坏预训练特征
    平衡(Batch BA 仅 +0.50) -> 加软启动 alpha(init 0.1, 前作 wavelet_scale 0.1
    的成功经验): x*(1+alpha*sigmoid(...)), 初始扰动~7%, 梯度仍非零、不可旁路。
    """

    def __init__(self, base_ratio: float = 0.1, soft_start: bool = False):
        super().__init__()
        self.base_ratio = base_ratio
        self.soft_start = soft_start
        self.w_s = nn.Parameter(torch.ones(1))   # 结构路权重(非零初始化)
        self.w_e = nn.Parameter(torch.ones(1))   # 边缘路权重
        self.alpha = nn.Parameter(torch.tensor(0.1)) if soft_start else None

    @staticmethod
    def _norm_map(t: torch.Tensor) -> torch.Tensor:
        """空间统计归一化 -> (0,1): (x-mean)/var 后 sigmoid(逐通道)。"""
        mu = t.mean(dim=[2, 3], keepdim=True)
        var = t.var(dim=[2, 3], keepdim=True)
        return torch.sigmoid((t - mu) / (var + _EPS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        in_dtype = x.dtype
        xf = x.float()
        freq = torch.fft.fftshift(torch.fft.fft2(xf, dim=(-2, -1)))
        mask = _gaussian_lowpass(h, w, x.device, torch.float32, self.base_ratio)
        low = torch.abs(torch.fft.ifft2(freq * mask, dim=(-2, -1)))
        high = torch.abs(torch.fft.ifft2(freq * (1 - mask), dim=(-2, -1)))
        s_map = self._norm_map(low ** 2)                     # 结构能量注意力
        e_map = self._norm_map((high - high.mean(dim=[2, 3], keepdim=True)) ** 2)  # 边缘方差注意力
        gate = torch.sigmoid(self.w_s * s_map + self.w_e * e_map)
        if self.alpha is not None:
            gate = self.alpha * gate
        out = xf * (1 + gate)
        return out.to(in_dtype)


class LSDG(nn.Module):
    """C: 局部统计方向门控卷积(H/V 条带边缘 + 3x3 局部信噪门控)。"""

    def __init__(self, dim: int, k: int = 7):
        super().__init__()
        self.strip_h = nn.Conv2d(dim, dim, (1, k), padding=(0, k // 2), groups=dim, bias=False)
        self.strip_v = nn.Conv2d(dim, dim, (k, 1), padding=(k // 2, 0), groups=dim, bias=False)
        self.fuse = nn.Conv2d(2 * dim, dim, 1, bias=False)
        nn.init.trunc_normal_(self.fuse.weight, std=0.02)    # 温和启动(非零, 梯度可流)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.fuse(torch.cat([self.strip_h(x), self.strip_v(x)], dim=1))
        a = e.abs()
        mu = F.avg_pool2d(a, 3, stride=1, padding=1)         # 3x3 局部统计(参数free门控)
        sq = F.avg_pool2d(a * a, 3, stride=1, padding=1)
        std = torch.sqrt(torch.clamp(sq - mu * mu, min=0) + _EPS)
        g = torch.sigmoid((a - mu) / (std + _EPS))           # 高于局部均值的响应才注入
        return x + e * g


class EPBlock(Block):
    """ConvNeXt Block + 可选 FESR(dwconv 后) / LSDG(dwconv 后)。

    dwconv/norm/pwconv/gamma/drop_path 的参数名与路径和原 Block 完全一致,
    预训练权重 100% 可加载; 新增段以 fesr_/lsdg_ 前缀进入白名单。
    """

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6,
                 use_fesr=False, use_lsdg=False, fesr_ratio=0.1, lsdg_k=7,
                 fesr_soft=False):
        super().__init__(dim, drop_rate, layer_scale_init_value)
        self.fesr = FESR(fesr_ratio, soft_start=fesr_soft) if use_fesr else None
        self.lsdg = LSDG(dim, lsdg_k) if use_lsdg else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        if self.lsdg is not None:
            x = self.lsdg(x)
        if self.fesr is not None:
            x = self.fesr(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXtEP(ConvNeXt):
    """ConvNeXt + {A:DPH head, B:FESR@stages, C:LSDG@stages} 结构化组合。"""

    def __init__(self, use_dph=False, fesr_stages=(), lsdg_stages=(), fesr_soft=False, **kw):
        super().__init__(**kw)
        dims0 = self.downsample_layers[0][0].out_channels
        self.dims = [dims0] + [self.downsample_layers[i][1].out_channels for i in (1, 2, 3)]

        for si in fesr_stages:
            self.stages[si] = nn.Sequential(
                *[EPBlock(self.dims[si], use_fesr=True, fesr_soft=fesr_soft)
                  for _ in self.stages[si]])
        for si in lsdg_stages:
            self.stages[si] = nn.Sequential(
                *[EPBlock(self.dims[si], use_lsdg=True) for _ in self.stages[si]])

        self.use_dph = use_dph
        if use_dph:
            d2, d3, d4 = self.dims[1], self.dims[2], self.dims[3]
            self.dph4_n = nn.LayerNorm(d4, eps=1e-6)            # s4 GMP 的 LN
            self.dph3_a = nn.Linear(d3, d4, bias=False)
            self.dph3_m = nn.Linear(d3, d4, bias=False)
            self.dph3_an = nn.LayerNorm(d4, eps=1e-6)
            self.dph3_mn = nn.LayerNorm(d4, eps=1e-6)
            self.dph2_a = nn.Linear(d2, d4, bias=False)
            self.dph2_m = nn.Linear(d2, d4, bias=False)
            self.dph2_an = nn.LayerNorm(d4, eps=1e-6)
            self.dph2_mn = nn.LayerNorm(d4, eps=1e-6)
            self.head = nn.Linear(6 * d4, self.head.out_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feats.append(x)
        gap4 = self.norm(feats[3].mean([-2, -1]))
        if not self.use_dph:
            return gap4
        gmp4 = self.dph4_n(feats[3].amax([-2, -1]))
        a3 = self.dph3_an(self.dph3_a(feats[2].mean([-2, -1])))
        m3 = self.dph3_mn(self.dph3_m(feats[2].amax([-2, -1])))
        a2 = self.dph2_an(self.dph2_a(feats[1].mean([-2, -1])))
        m2 = self.dph2_mn(self.dph2_m(feats[1].amax([-2, -1])))
        return torch.cat([gap4, gmp4, a3, m3, a2, m2], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


_TINY = dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])


def _ep(num_classes, **kw):
    return ConvNeXtEP(num_classes=num_classes, **{**_TINY, **kw})


def convnext_dph(num_classes: int):            # A 单独
    return _ep(num_classes, use_dph=True)

def convnext_fesr(num_classes: int):           # B 单独 (stage3/4)
    return _ep(num_classes, fesr_stages=(2, 3))

def convnext_fesr2(num_classes: int):          # B-v2 软启动(alpha init 0.1)
    return _ep(num_classes, fesr_stages=(2, 3), fesr_soft=True)

def convnext_lsdg(num_classes: int):           # C 单独 (stage2)
    return _ep(num_classes, lsdg_stages=(1,))

def convnext_dph_fesr(num_classes: int):       # A+B
    return _ep(num_classes, use_dph=True, fesr_stages=(2, 3))

def convnext_dph_lsdg(num_classes: int):       # A+C
    return _ep(num_classes, use_dph=True, lsdg_stages=(1,))

def convnext_fesr_lsdg(num_classes: int):      # B+C
    return _ep(num_classes, fesr_stages=(2, 3), lsdg_stages=(1,))

def convnext_dph_fesr_lsdg(num_classes: int):  # A+B+C 三合一(最终候选)
    return _ep(num_classes, use_dph=True, fesr_stages=(2, 3), lsdg_stages=(1,))

def convnext_epnet(num_classes: int):          # A+B(v2软启动)+C 三合一
    return _ep(num_classes, use_dph=True, fesr_stages=(2, 3), lsdg_stages=(1,),
               fesr_soft=True)
