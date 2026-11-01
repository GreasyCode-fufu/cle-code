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
    """C-v1: 局部统计方向门控卷积(残差注入式) — Batch BB 证实与 DPH 的 GMP 功能
    冗余(A+C < A), 且残差注入本质是可被优化器压制的叠加分支 -> 被 DSD 替换式设计取代,
    保留仅作消融对照。"""

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


class AEP(nn.Module):
    """自适应证据池化(Adaptive Evidence Pooling): softmax 加权池化谱。

    pool(x) = sum_i softmax(alpha*x_i) * x_i: alpha->0 退化为 GAP(全局统计),
    alpha->inf 逼近 GMP(最显著局部证据) —— 在"均值-最大"连续谱上为每个尺度
    学习最优插值点。支持正负响应(无需半波整流), 每尺度独立 alpha(init 1.0)。
    DPH(GAP/GMP 两端拼接)的数学深化: 拼接 -> 谱上自适应。"""

    def __init__(self, init_alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, C, H, W)
        w = torch.softmax(self.alpha * x.flatten(2), dim=-1)
        return (w * x.flatten(2)).sum(-1)


class CMPHead(nn.Module):
    """DPEA 创新点3: 通道级混合池化聚合头(Channel-wise Mixed Pooling)。

    每尺度 pooled_i = a_i ⊙ GAP_i + (1-a_i) ⊙ GMP_i, a_i 为逐通道可学习
    插值系数(sigmoid, init≈0.75 偏向 GAP 控方差), 三尺度投影拼接 -> 分类。
    数学上 DPH(GAP/GMP 两端硬拼接, 6*d4) 的连续松弛版: 每通道在"均值-最大"
    谱上自适应取点, 维度减 1/3(2304 vs 4608), 小样本下方差更小。
    吸收 DPH 验证过的 sens_M 增益证据(max 通道对恶性局部证据敏感)但受控。"""

    def __init__(self, dims, num_classes, init_bias: float = 1.1):
        super().__init__()
        d2, d3, d4 = dims[1], dims[2], dims[3]
        self.p3 = nn.Linear(d3, d4, bias=False)
        self.p2 = nn.Linear(d2, d4, bias=False)
        self.n4 = nn.LayerNorm(d4, eps=1e-6)
        self.n3 = nn.LayerNorm(d4, eps=1e-6)
        self.n2 = nn.LayerNorm(d4, eps=1e-6)
        self.a4 = nn.Parameter(torch.full((d4,), init_bias))   # sigmoid(1.1)≈0.75
        self.a3 = nn.Parameter(torch.full((d3,), init_bias))   # 插值在原尺度通道上, 投影后
        self.a2 = nn.Parameter(torch.full((d2,), init_bias))
        self.head = nn.Linear(3 * d4, num_classes)

    def _mix(self, t):
        gap = t.mean([-2, -1])
        gmp = t.amax([-2, -1])
        return gap, gmp

    def forward(self, feats):
        gap4, gmp4 = self._mix(feats[3])
        gap3, gmp3 = self._mix(feats[2])
        gap2, gmp2 = self._mix(feats[1])
        e4 = self.n4(torch.sigmoid(self.a4) * gap4 + (1 - torch.sigmoid(self.a4)) * gmp4)
        e3 = self.n3(self.p3(torch.sigmoid(self.a3) * gap3 + (1 - torch.sigmoid(self.a3)) * gmp3))
        e2 = self.n2(self.p2(torch.sigmoid(self.a2) * gap2 + (1 - torch.sigmoid(self.a2)) * gmp2))
        return self.head(torch.cat([e4, e3, e2], dim=-1))


class ArcMarginHead(nn.Module):
    """D: 判别性角度间隔头(ArcFace 思想融入分类网络, 网络结构+损失一体创新)。

    DPH 拼接特征 -> Linear 降维 -> LN -> L2 归一化 -> 与类中心做角度间隔判别。
    训练: s*cos(theta + m) 强制类间角间隔(m=0.3); 推理: s*cos(theta) 前向自洽。
    动机: M/B 混淆(前作 8 例恶性漏诊)源于决策边界贴合, 角度间隔在归一化空间
    强制 M/B 分离; 特征归一化同时消除类内幅值漂移(小样本稳健)。
    """

    def __init__(self, feat_dim: int, num_classes: int, emb_dim: int = 256,
                 s: float = 16.0, m: float = 0.3):
        super().__init__()
        self.fc = nn.Linear(feat_dim, emb_dim, bias=False)
        self.ln = nn.LayerNorm(emb_dim, eps=1e-6)
        self.weight = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.normal_(self.weight, std=0.01)
        self.s, self.m = s, m

    def forward(self, feat: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        x = F.normalize(self.ln(self.fc(feat)), dim=-1)
        w = F.normalize(self.weight, dim=-1)
        cos = x @ w.t()                                     # (B, nc) in [-1, 1]
        if self.training:
            onehot = torch.zeros_like(cos)
            if labels is not None:
                onehot.scatter_(1, labels.view(-1, 1).to(cos.device), 1.0)
            else:                                           # 无标签时退化为伪标签
                onehot.scatter_(1, cos.argmax(dim=1, keepdim=True), 1.0)
            theta = torch.acos(torch.clamp(cos, -1 + 1e-7, 1 - 1e-7))
            target_cos = torch.cos(theta + self.m * onehot)
            cos = cos * (1 - onehot) + target_cos * onehot
        return self.s * cos


class MFM(nn.Module):
    """E: 掩码特征重建自监督辅助头(Masked Feature Modeling)。

    training 时对 stage3 特征(14x14)随机 mask 25% patch, 轻量 decoder 重建
    (仅在被 mask 位置计 L2); 推理零开销。损失经 model.mfm_loss 由训练循环聚合
    (CE + lambda*MFM, lambda=0.1)。动机: 2k 小样本下迫使网络编码完整腺体结构
    语义而非捷径特征(自监督正则)。
    """

    def __init__(self, dim: int, mask_ratio: float = 0.25):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.fc = nn.Linear(dim, dim)
        self.loss = torch.zeros(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        b, c, h, w = x.shape
        n = h * w
        k = max(1, int(n * self.mask_ratio))
        idx = torch.randperm(n, device=x.device)[:k]
        flat = x.reshape(b, c, n)
        xm = flat.clone()
        xm[:, :, idx] = 0                                   # patch 级掩码
        inp = xm.permute(0, 2, 1).reshape(b * n, c)         # 逐位置特征
        rec = self.fc(self.ln(inp))                         # 轻量 decoder
        rec = rec.reshape(b, n, c).permute(0, 2, 1)         # (b, c, n)
        self.loss = F.mse_loss(rec[:, :, idx].float(),
                               flat[:, :, idx].detach().float())
        return x


class DSDConv(nn.Module):
    """C-v2: 方向条带深度卷积 —— 替换式(非叠加)重设计, 2026-11-01。

    7×7 dwconv 整体替换为 (7×1)+(1×7) 并联条带深度卷积 + 通道级 softmax 混合。
    数学动机: CLE 胃体腺体呈方向性管状结构, 条带核以 14C 参数(C,1,7,7) 的 29%)
    显式编码 H/V 方向感受野; 全核是条带复合的严格超集, 属算子子空间替换。
    预训练投影初始化: K(u,v) -> [Σ_v K (7,1); Σ_u K (1,7)] (幅值守恒投影),
    使替换后初始响应与原 7×7 核统计一致, 预训练语义无损迁移。
    """

    def __init__(self, dim: int, k: int = 7):
        super().__init__()
        self.dsv = nn.Conv2d(dim, dim, (k, 1), padding=(k // 2, 0), groups=dim, bias=True)
        self.dsh = nn.Conv2d(dim, dim, (1, k), padding=(0, k // 2), groups=dim, bias=False)
        self.dsm = nn.Parameter(torch.zeros(2, dim, 1, 1))   # softmax -> [0.5, 0.5]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.dsm, dim=0)
        return w[0] * self.dsv(x) + w[1] * self.dsh(x)


class FGC(nn.Module):
    """B-v2: 频域全局卷积 —— 替换式重设计(GFNet 思想 + 预训练核频域等价初始化)。

    stage4 的 7×7 dwconv(7×7 特征图) 整体替换为 rfft2 频域乘性滤波(循环卷积的频域对偶)。
    初始化 = rfft2(roll(预训练核, -3)): 初始行为原 dwconv 的循环卷积近似,
    训练中感受野可自由从 7×7 扩展到全图(全局频率调制), 参数 37.7K->86K。
    依据: M→B 漏诊样本高频占比低 15%, 需要超局部核的全局频率重组能力。
    """

    def __init__(self, dim: int, hw: int = 7, k: int = 7):
        super().__init__()
        self.hw = hw
        self.flt = nn.Parameter(torch.zeros(dim, hw, hw // 2 + 1, 2))   # 实部/虚部
        self.flt_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert h == self.hw and w == self.hw, f"FGC 需要 {self.hw}x{self.hw} 输入, got {h}x{w}"
        in_dtype = x.dtype
        xf = x.float()
        f = torch.view_as_complex(self.flt)
        y = torch.fft.irfft2(torch.fft.rfft2(xf, dim=(-2, -1)) * f, s=(h, w), dim=(-2, -1))
        return (y + self.flt_bias.view(1, -1, 1, 1)).to(in_dtype)


class EPBlock(Block):
    """ConvNeXt Block + 可选 FESR(dwconv 后) / LSDG(dwconv 后)。

    dwconv/norm/pwconv/gamma/drop_path 的参数名与路径和原 Block 完全一致,
    预训练权重 100% 可加载; 新增段以 fesr_/lsdg_ 前缀进入白名单。
    """

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6,
                 use_fesr=False, use_lsdg=False, fesr_ratio=0.1, lsdg_k=7,
                 fesr_soft=False, use_dsd=False, dsd_k=7, use_fgc=False, fgc_hw=14):
        super().__init__(dim, drop_rate, layer_scale_init_value)
        self.fesr = FESR(fesr_ratio, soft_start=fesr_soft) if use_fesr else None
        self.lsdg = LSDG(dim, lsdg_k) if use_lsdg else None
        if use_dsd:
            self.dwconv = DSDConv(dim, dsd_k)
        if use_fgc:
            self.dwconv = FGC(dim, fgc_hw)

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

    def __init__(self, use_dph=False, fesr_stages=(), lsdg_stages=(), fesr_soft=False,
                 dsd_stages=(), fgc_stages=(), use_am=False, use_mfm=False,
                 use_aep=False, am_aux=False, mfm_lam=0.05, mfm_mask=0.15,
                 use_dph4=False, use_cmp=False, **kw):
        super().__init__(**kw)
        dims0 = self.downsample_layers[0][0].out_channels
        self.dims = [dims0] + [self.downsample_layers[i][1].out_channels for i in (1, 2, 3)]

        for si in fesr_stages:
            self.stages[si] = nn.Sequential(
                *[EPBlock(self.dims[si], use_fesr=True, fesr_soft=fesr_soft)
                  for _ in self.stages[si]])
        for si in dsd_stages:
            self.stages[si] = nn.Sequential(
                *[EPBlock(self.dims[si], use_dsd=True) for _ in self.stages[si]])
        for si in fgc_stages:
            self.stages[si] = nn.Sequential(
                *[EPBlock(self.dims[si], use_fgc=True,
                          fgc_hw={0: 56, 1: 28, 2: 14, 3: 7}.get(si, 7))
                  for _ in self.stages[si]])
        for si in lsdg_stages:
            self.stages[si] = nn.Sequential(
                *[EPBlock(self.dims[si], use_lsdg=True) for _ in self.stages[si]])

        self.use_dph = use_dph
        self.use_am = use_am
        self.use_aep = use_aep
        self.use_dph4 = use_dph4
        if use_cmp:
            self.cmp_head = CMPHead(self.dims, self.head.out_features)
            self.head = nn.Identity()
        self.forward_labels = use_am or am_aux   # AMH 需要标签(主头或辅助模式)
        if use_mfm:
            self.mfm = MFM(self.dims[2], mask_ratio=mfm_mask)
            self.mfm_lambda = mfm_lam
        if use_aep:
            d4_ = self.dims[3]
            self.aep3 = AEP(); self.aep2 = AEP()      # s3/s2 尺度的自适应谱池化
            self.aep3_n = nn.LayerNorm(d4_, eps=1e-6)
            self.aep2_n = nn.LayerNorm(d4_, eps=1e-6)
            self.aep3_p = nn.Linear(self.dims[2], d4_, bias=False)
            self.aep2_p = nn.Linear(self.dims[1], d4_, bias=False)
        if am_aux:                                     # AMH 辅助正则(主 head 不变)
            d4_ = self.dims[3]
            self.am_aux = ArcMarginHead(6 * d4_ if use_dph else d4_,
                                        self.head.out_features)
        if use_dph:
            d2, d3, d4 = self.dims[1], self.dims[2], self.dims[3]
            self.dph4_n = nn.LayerNorm(d4, eps=1e-6)            # s4 GMP 的 LN
            if use_aep:   # s2/s3 走 AEP 谱池化, 不创建双池化投影
                self.head = nn.Linear(4 * d4, self.head.out_features)
            elif use_dph4:  # DPH-lite: GMP 仅语义层 s4, s2/s3 仅 GAP 投影
                self.dph3_a = nn.Linear(d3, d4, bias=False)
                self.dph3_an = nn.LayerNorm(d4, eps=1e-6)
                self.dph2_a = nn.Linear(d2, d4, bias=False)
                self.dph2_an = nn.LayerNorm(d4, eps=1e-6)
                self.head = nn.Linear(4 * d4, self.head.out_features)
            else:
                self.dph3_a = nn.Linear(d3, d4, bias=False)
                self.dph3_m = nn.Linear(d3, d4, bias=False)
                self.dph3_an = nn.LayerNorm(d4, eps=1e-6)
                self.dph3_mn = nn.LayerNorm(d4, eps=1e-6)
                self.dph2_a = nn.Linear(d2, d4, bias=False)
                self.dph2_m = nn.Linear(d2, d4, bias=False)
                self.dph2_an = nn.LayerNorm(d4, eps=1e-6)
                self.dph2_mn = nn.LayerNorm(d4, eps=1e-6)
                if use_am:
                    self.head = ArcMarginHead(6 * d4, self.head.out_features)
                else:
                    self.head = nn.Linear(6 * d4, self.head.out_features)
        elif use_am:                        # 无 DPH 时基于 s4 单尺度
            self.head = ArcMarginHead(d4, self.head.out_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 2 and getattr(self, "mfm", None) is not None:
                x = self.mfm(x)                             # 训练时掩码重建(返回原x)
            feats.append(x)
        gap4 = self.norm(feats[3].mean([-2, -1]))
        if not self.use_dph:
            return gap4
        gmp4 = self.dph4_n(feats[3].amax([-2, -1]))
        if getattr(self, "use_dph4", False):   # DPH-lite: GMP 仅 s4
            a3 = self.dph3_an(self.dph3_a(feats[2].mean([-2, -1])))
            a2 = self.dph2_an(self.dph2_a(feats[1].mean([-2, -1])))
            return torch.cat([gap4, gmp4, a3, a2], dim=-1)   # 4*768
        if self.use_aep:   # s2/s3: 双池化拼接 -> 自适应谱池化(参数更少, 尺度自适应)
            e3 = self.aep3_n(self.aep3_p(self.aep3(feats[2])))
            e2 = self.aep2_n(self.aep2_p(self.aep2(feats[1])))
            return torch.cat([gap4, gmp4, e3, e2], dim=-1)   # 4*768
        a3 = self.dph3_an(self.dph3_a(feats[2].mean([-2, -1])))
        m3 = self.dph3_mn(self.dph3_m(feats[2].amax([-2, -1])))
        a2 = self.dph2_an(self.dph2_a(feats[1].mean([-2, -1])))
        m2 = self.dph2_mn(self.dph2_m(feats[1].amax([-2, -1])))
        return torch.cat([gap4, gmp4, a3, m3, a2, m2], dim=-1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        if getattr(self, "cmp_head", None) is not None:
            feats = []
            xx = x
            for i in range(4):
                xx = self.downsample_layers[i](xx)
                xx = self.stages[i](xx)
                feats.append(xx)
            return self.cmp_head(feats)               # cmp 内部含分类层
        f = self.forward_features(x)
        if getattr(self, "am_aux", None) is not None and self.training:
            self.aux_logits = self.am_aux(f, labels if labels is not None else None)
        if self.use_am and self.training and labels is not None:
            return self.head(f, labels)      # 目标类角度间隔(见 ArcMarginHead.forward)
        return self.head(f)


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

def convnext_epnet(num_classes: int):          # A+B(v2软启动)+C(v1) 三合一(旧链)
    return _ep(num_classes, use_dph=True, fesr_stages=(2, 3), lsdg_stages=(1,),
               fesr_soft=True)

# ── 算子替换式变体 (2026-11-01 重设计: 替换 > 叠加) ──
def convnext_dsd(num_classes: int):            # C-v2: DSD 条带卷积替换 stage2 dwconv
    return _ep(num_classes, dsd_stages=(1,))

def convnext_fgc(num_classes: int):            # B-v2: FGC 频域全局卷积替换 stage4 dwconv
    return _ep(num_classes, fgc_stages=(3,))

def convnext_dph_dsd(num_classes: int):        # A + C-v2
    return _ep(num_classes, use_dph=True, dsd_stages=(1,))

def convnext_dph_fgc(num_classes: int):        # A + B-v2
    return _ep(num_classes, use_dph=True, fgc_stages=(3,))

def convnext_fgc_dsd(num_classes: int):        # B-v2 + C-v2
    return _ep(num_classes, fgc_stages=(3,), dsd_stages=(1,))

def convnext_epnet2(num_classes: int):         # A + B-v2 + C-v2 (替换式三合一)
    return _ep(num_classes, use_dph=True, fgc_stages=(3,), dsd_stages=(1,))

def convnext_dph_fesr2(num_classes: int):      # A + B'(软启动FESR) 组合
    return _ep(num_classes, use_dph=True, fesr_stages=(2, 3), fesr_soft=True)


# ── 损失/自监督增强变体 (2026-11-01: loss+SSL 正交于结构, 无组合回退) ──
def convnext_dph_am(num_classes: int):       # A + D(AMH 角度间隔头)
    return _ep(num_classes, use_dph=True, use_am=True)

def convnext_dph_mfm(num_classes: int):      # A + E(MFM 掩码特征重建自监督)
    return _ep(num_classes, use_dph=True, use_mfm=True)


def convnext_dph_am2(num_classes: int):      # A + AMH辅助正则(主head不变, 零瓶颈)
    return _ep(num_classes, use_dph=True, am_aux=True)

def convnext_dph_mfm2(num_classes: int):     # A + MFM温和化(λ=0.05, mask15%)
    return _ep(num_classes, use_dph=True, use_mfm=True, mfm_lam=0.05, mfm_mask=0.15)

def convnext_aep(num_classes: int):          # A-v3: s4双池 + s2/s3自适应谱池化
    return _ep(num_classes, use_dph=True, use_aep=True)


def convnext_dph4(num_classes: int):         # A-lite: GMP 仅 s4(语义层), s2/s3 GAP
    return _ep(num_classes, use_dph=True, use_dph4=True)


# ── DPEA-Net: 双路证据提取+自适应聚合 (2026-11-02 重设计) ──
def convnext_cmp(num_classes: int):          # 创新点3单独: CMP 聚合头
    return _ep(num_classes, use_cmp=True)

def convnext_fse_dsd(num_classes: int):      # 创新1+2: 频域路+空间路
    return _ep(num_classes, fesr_stages=(2, 3), fesr_soft=True, dsd_stages=(1,))

def convnext_dpea(num_classes: int):         # DPEA-Net 三合一(最终网络候选)
    return _ep(num_classes, fesr_stages=(2, 3), fesr_soft=True, dsd_stages=(1,),
               use_cmp=True)


def convnext_cmp_fgc(num_classes: int):     # CMP + FGC (聚合头+频域算子)
    return _ep(num_classes, use_cmp=True, fgc_stages=(3,))

def convnext_cmp_dsd(num_classes: int):     # CMP + DSD (聚合头+方向算子)
    return _ep(num_classes, use_cmp=True, dsd_stages=(1,))

def convnext_cmp_fse(num_classes: int):     # CMP + FSE (聚合头+频域重标定)
    return _ep(num_classes, use_cmp=True, fesr_stages=(2, 3), fesr_soft=True)
