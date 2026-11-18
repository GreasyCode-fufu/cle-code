"""CLE-Net: ConvNeXt-T 三创新点改进 —— 2026-10-25 v2 设计

与第一代(convnext_cle.py, 小波家族 WDS/WTB/MSF/WDD)完全正交的三条轴线:

  A. EGB Edge-Gated Block    —— 块内: PFESA(MICCAI2026, 无参数FFT边缘-结构注意力)
       作为零参数先验门控分支, 调制 dwconv 输出。
       动机: CLE 高频能量占85%+, 低对比度下需可学习区分"细胞边缘"vs"荧光噪声";
       零参数注意力避免小数据集过拟合(SE/CBAM 之弊)。
  B. SCG Strip Context Guidance —— stage3/4 块内: StripConv(AAAI2026, 1x19/19x1
       条带卷积)零门控并联分支, 捕获腺体隐窝的各向异性长程结构。
       动机: 腺体圆/卵圆形态判别需要超出 7x7 的条带上下文。
  C. HSC Head with Synergistic Calibration —— 分类头前: SCSA(Arxiv2024) 通道
       自注意力核心的轻量版(PCSA), 对 GAP 特征做空间-通道协同重标定。
       动机: benign vs malignant 腺体形态差异细微且 Malignant 样本最少,
       通道重标定 = 软特征选择。

工程约束(与 train_cle_fair 严格加载器协同, 承袭已验证模式):
  - 原始 ConvNeXt 全部参数路径不变; ImageNet ema 权重 100% 加载;
  - 新增参数段: egb_g / scg / scg_g / hsc, 需加白名单;
  - 全部新分支零初始化门控 => 初始严格等价于原 ConvNeXt。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Convnext import ConvNeXt, Block

# ═══════════════ HSD: 病理亚型 -> 临床大类 的固定聚合矩阵 ═══════════════
# 亚型顺序(与文件名前缀一致, 见 analysis/subtype_error.py 统计):
#   0 atrophy(萎缩)  1 intestinal_metaplasia(肠化)  2 normal(正常)
#   3 dysplasia(异型增生)  4 cancer(癌)
# 临床大类顺序(与 class_indices.json 一致): 0 Benign 1 Malignant 2 Normal
# 依据 Correa 级联: 正常 -> 萎缩/肠化(良性) -> 异型增生/癌(恶性)
SUBTYPE_NAMES = ["atrophy", "intestinal_metaplasia", "normal", "dysplasia", "cancer"]
HSD_AGGREGATE = [
    [1.0, 1.0, 0.0, 0.0, 0.0],   # Benign    <- atrophy, intestinal_metaplasia
    [0.0, 0.0, 0.0, 1.0, 1.0],   # Malignant <- dysplasia, cancer
    [0.0, 0.0, 1.0, 0.0, 0.0],   # Normal    <- normal
]


# ═══════════════ A: PFESA 核心(内联精简, 无参数) ═══════════════

class _PFECore(nn.Module):
    """PFESA 的边缘+结构双注意力核心。无参数、逐位置门控, 输出 0~1 权重图。"""
    def __init__(self, base_ratio=0.1):
        super().__init__()
        self.base_ratio = base_ratio
        self.eps = 1e-5

    def _low_mask(self, h, w, device):
        r = self.base_ratio * min(h, w) / max(h, w)
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        return torch.exp(-(gy ** 2 + gx ** 2) / (2 * r ** 2))  # (H,W)

    def forward(self, feat):
        b, c, h, w = feat.shape
        ff = torch.fft.fftshift(torch.fft.fftn(feat.float(), dim=(-2, -1)), dim=(-2, -1))
        m = self._low_mask(h, w, feat.device)                   # (H,W)
        low = torch.abs(torch.fft.ifftn(torch.fft.ifftshift(ff * m, dim=(-2, -1)), dim=(-2, -1)))
        high = torch.abs(torch.fft.ifftn(torch.fft.ifftshift(ff * (1 - m), dim=(-2, -1)), dim=(-2, -1)))
        # 低频结构注意力: 能量归一化
        le = low.pow(2)
        sa = torch.sigmoid((le - le.mean(dim=[2, 3], keepdim=True)) /
                           (le.var(dim=[2, 3], keepdim=True) + self.eps))
        # 高频边缘注意力: 方差归一的局部突变度
        hm = high.mean(dim=[2, 3], keepdim=True)
        ea = (high - hm).pow(2) / (high.var(dim=[2, 3], keepdim=True) + self.eps)
        return torch.sigmoid(sa + ea)                            # (B,C,H,W) 0~1


class EGBBlock(Block):
    """ConvNeXt Block + PFESA 零参数门控分支(作用在 dwconv 输出上)。"""
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6, base_ratio=0.1):
        super().__init__(dim, drop_rate, layer_scale_init_value)
        self.egb = _PFECore(base_ratio)
        self.egb_g = nn.Parameter(torch.zeros(1))               # 标量门: 0 => 初始等价

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x + self.egb_g * self.egb(x) * x                    # 门控自调制
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)




# ═══════════════ FBE: 频带均衡(Frequency Band Equalization) ═══════════════

class _FBECore(nn.Module):
    """FBE 核心: rfft2 径向 K 带分解(不含 DC), 每通道每带一个零初始化增益。

    与 EGB 的互补关系: EGB 回答"边缘在**哪里**"(逐位置门控, stage1-3);
    FBE 回答"纹理的**哪些频带**定义类别"(逐通道频带均衡, stage0)。
    依据: FOV 内频谱实测 Malignant band1 纹理占比 0.0298 vs Normal 0.0200(~1.4σ)。

    增益为静态(与输入无关): 小数据集上动态注意力(HSC)已实测过拟合(-0.64pp),
    静态均衡器仅 96x3=288 参数, 与"零参数 EGB"的设计语言一致。
    """
    def __init__(self, dim, n_bands=3):
        super().__init__()
        self.n_bands = n_bands
        self.fbe_w = nn.Parameter(torch.zeros(dim, n_bands))   # [C,K] 零初始化 => 恒等

    def forward(self, x):
        b, c, h, w = x.shape
        xf = torch.fft.rfft2(x.float())
        hf, wf = xf.shape[-2:]
        fy = torch.fft.fftfreq(hf, device=x.device) * 2.0          # 归一化 -1..1
        fx = torch.arange(wf, device=x.device).float() * (2.0 / max(float(h), 1.0))
        r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        out = None
        for k, (lo, hi) in enumerate([(0.0, 0.35), (0.35, 0.70), (0.70, 1e9)]):
            m = (r > lo) & (r <= hi)
            if not bool(m.any()):
                continue
            comp = torch.fft.irfft2(xf * m.to(xf.dtype), s=(h, w))  # 该带空域分量
            term = self.fbe_w[:, k].view(1, c, 1, 1) * comp
            out = term if out is None else out + term
        if out is None:
            return torch.zeros_like(x)
        return out.to(x.dtype)


# ═══════════════ SR: 频域辅助正则化分支(训练期专用, 推理路径不经过) ═══════════════

class SRBranch(nn.Module):
    """Spectral Regularisation 辅助分支: 对预分类头特征 f 做 DCT 频带能量约束。

    数据流: f[B,768] -> Linear(768->512, 无bias) -> LayerNorm(512, 无affine)
            -> 正交归一化 DCT-II(沿特征维, fp32) -> 三频带平均平方能量 E[B,3]。

    设计约束(见 experiments/SR_DESIGN.md):
      - LN 无 affine => 总谱能量恒为 512 且 DC=0(Parseval), 从构造上排除
        "无界放大特征幅值来满足损失"的退化解; 能量份额 share=E/sum(E) 落在单纯形;
      - 仅在 model.training 时由 forward 计算(经 self._sr_bands 供训练脚本读取),
        eval/推理路径完全不经过本分支, 推理零开销;
      - proj 属骨干参数组(lr 1e-4, wd 5e-2): FBE 的教训是 lr×100/wd0 的新增增益
        会在小数据集上膨胀成纯过拟合容量。
    """
    def __init__(self, dim, sr_dim=512):
        super().__init__()
        self.sr_dim = sr_dim
        self.proj = nn.Linear(dim, sr_dim, bias=False)
        n = torch.arange(sr_dim, dtype=torch.float64)
        k = torch.arange(sr_dim, dtype=torch.float64).view(-1, 1)
        a = torch.full((sr_dim, 1), (2.0 / sr_dim) ** 0.5, dtype=torch.float64)
        a[0, 0] = (1.0 / sr_dim) ** 0.5
        self.register_buffer("dct", (a * torch.cos(torch.pi * (2 * n + 1) * k / (2 * sr_dim))).float())
        self.bands = [(0, sr_dim // 4), (sr_dim // 4, sr_dim // 2), (sr_dim // 2, sr_dim)]

    def forward(self, f):
        z = F.layer_norm(self.proj(f), (self.sr_dim,))
        spec = (z.float() @ self.dct.T).pow(2)                 # fp32, 与 EGB 的 FFT 同策略
        return torch.stack([spec[:, a:b].mean(dim=1) for a, b in self.bands], dim=1)  # [B,3]


# ═══════════════ B: StripConv 门控并联分支 ═══════════════

class StripBranch(nn.Module):
    """AAAI2026 StripConv 的并联分支形态: 5x5 dw + 1x19 + 19x1 dw + 1x1, 输出残差图。

    死锁修复(见 GATE_DIAGNOSIS.md): 末层 c3 零初始化 => 初始 S(x)=0, 分支与骨干
    严格恒等; 但 c3 的梯度 = dL/dy @ (c2c1c0(x))^T 非零, 故 c3 先被训练, 内部层随后
    解锁(ReZero/ControlNet 式渐进解锁), 不再依赖一个会被 wd 按死的外部标量门。
    """
    def __init__(self, dim, k2=19):
        super().__init__()
        self.c0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.c1 = nn.Conv2d(dim, dim, (1, k2), padding=(0, k2 // 2), groups=dim)
        self.c2 = nn.Conv2d(dim, dim, (k2, 1), padding=(k2 // 2, 0), groups=dim)
        self.c3 = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)

    def forward(self, x):
        a = self.c3(self.c2(self.c1(self.c0(x))))
        return a


class SCGBlock(Block):
    """ConvNeXt Block + StripConv 零初始化残差分支(作用在 dwconv 输出后)。"""
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6, k2=19):
        super().__init__(dim, drop_rate, layer_scale_init_value)
        self.scg = StripBranch(dim, k2)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x + self.scg(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


# ═══════════════ C: HSC 通道重标定头(SE 式, 末层零初始化) ═══════════════

class HSCHead(nn.Module):
    """GAP 后的通道重标定头(通道软特征选择)。

    修复(见 GATE_DIAGNOSIS.md Q02): 旧实现把 768 维 reshape 成 (8,96) 后做
    q@k^T, 得到的是 (B,8,8) 的"头间"注意力, 既非通道注意力也非标准 MSA, 且经
    g≈0 标量门死锁。现改为 SE 式通道门控:
      z = tanh( W2 · GELU(W1 · f) ),  W2 零初始化 => 初始 z=0, f'=f 严格恒等;
      f' = f ⊙ (1+z),  z∈(-1,1) 为逐通道有界校正, 等价于软特征选择。
    零初始化的是末层 W2 而非外加标量门 => fc2 的梯度 = f⊙sech^2(0)⊙GELU(fc1·f)^T
    非零, 可正常训练, 不存在死锁。
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):                                       # x: (B, C)
        z = torch.tanh(self.fc2(F.gelu(self.fc1(x))))
        return x * (1.0 + z)


# ═══════════════ 统一叠加块(EGB/SCG 可同块共存) ═══════════════

class CLENetBlock(Block):
    """ConvNeXt Block, 可在同一块内同时承载 EGB(乘性边缘门控) 与 SCG(零初始化条带残差)。

    修复 _swap 覆盖 bug(GATE_DIAGNOSIS.md Q01): 旧 CLENet 用两次整 stage 替换, 后装的
    SCGBlock 把同 stage 的 EGBBlock 覆盖, 导致组合网络中 EGB/ESA 实际只剩 stage1。
    现改为单一块类按标志位叠加, stage2/3 可同时持有 EGB+SCG, 组合=各单点的真叠加。
    """
    def __init__(self, dim, use_egb=False, use_scg=False, use_fbe=False, drop_rate=0.,
                 layer_scale_init_value=1e-6, base_ratio=0.1, k2=19):
        super().__init__(dim, drop_rate, layer_scale_init_value)
        self.use_egb = use_egb
        self.use_scg = use_scg
        self.use_fbe = use_fbe
        if use_egb:
            self.egb = _PFECore(base_ratio)
            self.egb_g = nn.Parameter(torch.zeros(1))           # 标量门: 高 lr 组训练
        if use_scg:
            self.scg = StripBranch(dim, k2)                     # c3 零初始化 => 恒等
        if use_fbe:
            self.fbe = _FBECore(dim)                            # fbe_w 零初始化 => 恒等

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        if self.use_egb:
            x = x + self.egb_g * self.egb(x) * x                # EGB 乘性自调制
        if self.use_scg:
            x = x + self.scg(x)                                 # SCG 加性残差
        if self.use_fbe:
            x = x + self.fbe(x)                                 # FBE 频带均衡残差
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


# ═══════════════ 组合网络 ═══════════════

class MSFHead(nn.Module):
    """多尺度融合头 (Multi-Scale Fusion head, MSF)。

    输入: stage3 与 stage4 的 GAP 特征拼接(d3+d4); 输出: 与分类头同维的 logits 增量。
    输出层零初始化 => 训练起始时该分支严格为 0(等价于基线), 且"把输出层置零"即
    等价于"关闭该模块"(off == orig), 便于做 off/orig 干预证据。

    与 HSC 的本质区别: HSC 对**同一个**最终特征做通道重加权, 没有引入新信息,
    实测在组合中为负贡献(-0.64pp); MSF 引入 stage3 这一**新的**尺度信息,
    属于"增加信息"而非"重排已有信息"。
    """
    def __init__(self, d3, d4, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d3 + d4, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return self.net(z)


class CLENet(ConvNeXt):
    """ConvNeXt + {A:EGB@stages, B:SCG@stages, C:HSC} 渐进叠加。"""
    def __init__(self, egb_stages=(), scg_stages=(), fbe_stages=(), use_hsc=False,
                 use_msf=False, use_hsd=False, use_hsd_gate=False,
                 use_sr=False, sr_dim=512,
                 strip_k2=19, pfe_ratio=0.1, **kw):
        super().__init__(**kw)
        dims = [dl[1].out_channels if i > 0 else dl[0].out_channels
                for i, dl in enumerate(self.downsample_layers)]

        for si in sorted(set(egb_stages) | set(scg_stages) | set(fbe_stages)):
            news = []
            for blk in self.stages[si]:
                nb = CLENetBlock(dim=dims[si],
                                 use_egb=(si in egb_stages),
                                 use_scg=(si in scg_stages),
                                 use_fbe=(si in fbe_stages),
                                 base_ratio=pfe_ratio, k2=strip_k2)
                nb.load_state_dict(blk.state_dict(), strict=False)
                news.append(nb)
            self.stages[si] = nn.Sequential(*news)

        self.use_hsc = use_hsc
        if use_hsc:
            self.hsc = HSCHead(dims[-1])

        self.use_msf = use_msf
        if use_msf:
            self.msf = MSFHead(dims[2], dims[3], self.head.out_features)

        self.use_hsd = use_hsd
        if use_hsd:
            # HSD: 亚型层 -> 固定病理聚合矩阵 A -> 大类。原分类头退化为零初始化残差,
            # 因此训练起步时大类 logits 完全由亚型聚合产生(结构上的层次分解),
            # 残差仅在亚型聚合不足以表达时才被学习出来。
            self.hsd_sub = nn.Linear(dims[-1], len(SUBTYPE_NAMES))
            self.register_buffer("hsd_A", torch.tensor(HSD_AGGREGATE, dtype=torch.float32))
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            self._sub_logits = None

        self.use_sr = use_sr
        if use_sr:
            self.sr = SRBranch(dims[-1], sr_dim)
            self._sr_bands = None

        self.use_hsd_gate = use_hsd_gate
        if use_hsd_gate:
            # HSD-G: 主 logits 仍是原分类头(判别力不被强制聚合削弱), 亚型聚合项作为
            # 零初始化门控的病理先验修正。g=0 时与基线严格等价。
            self.hsd_sub = nn.Linear(dims[-1], len(SUBTYPE_NAMES))
            self.register_buffer("hsd_A", torch.tensor(HSD_AGGREGATE, dtype=torch.float32))
            self.hsd_g = nn.Parameter(torch.zeros(1))
            self._sub_logits = None

    def forward_features(self, x):
        gaps = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            gaps.append(x.mean([-2, -1]))                       # 各 stage 的 GAP
        self._gaps = gaps
        x = self.norm(gaps[-1])                                  # 末层 GAP (B,C)
        if self.use_hsc:
            x = self.hsc(x)
        return x

    def forward(self, x):
        f = self.forward_features(x)
        if self.use_sr and self.training:
            # 仅训练期计算频带能量(供训练脚本的 CSR/SPR 损失读取);
            # eval/推理路径不经过, 分类 logits 与不启用 SR 时完全一致。
            self._sr_bands = self.sr(f)                          # [B,3]
        if self.use_hsd:
            s = self.hsd_sub(f)
            self._sub_logits = s                       # 供辅助监督读取
            logits = s @ self.hsd_A.t() + self.head(f)  # 固定聚合 + 零初始化残差
        elif self.use_hsd_gate:
            s = self.hsd_sub(f)
            self._sub_logits = s
            logits = self.head(f) + self.hsd_g * (s @ self.hsd_A.t())
        else:
            logits = self.head(f)
        if self.use_msf:
            logits = logits + self.msf(torch.cat([self._gaps[2], self._gaps[3]], dim=1))
        return logits


def _tiny(nc, **kw):
    return CLENet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=nc, **kw)


def convnext_egb(num_classes: int):
    return _tiny(num_classes, egb_stages=(1, 2, 3))

def convnext_scg(num_classes: int):
    return _tiny(num_classes, scg_stages=(2, 3))

def convnext_hsc(num_classes: int):
    return _tiny(num_classes, use_hsc=True)

def convnext_scg_hsc(num_classes: int):
    """CLE-Net 形态: baseline + B(SCG) + C(HSC), 不含 EGB(两代实验证其 test 端拖累)"""
    return _tiny(num_classes, scg_stages=(2, 3), use_hsc=True)


def convnext_egb_scg(num_classes: int):
    return _tiny(num_classes, egb_stages=(1, 2, 3), scg_stages=(2, 3))


# ═══════════════ 主干容量缩放: CLE-Net 的 EGB+SCG 配方移植到 S/B 底座 ═══════════════
# 科学问题: EGB(stage1-3)/SCG(stage2-3) 的增益是否随主干容量缩放。
# 结构与 tiny 版完全同配方: EGB 乘性门控作用于 dwconv 后, SCG 条带残差并联,
# 均不改变维度/分辨率; 零初始化等价性同 tiny(冒烟断言覆盖)。

def _small(nc, **kw):
    return CLENet(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], num_classes=nc, **kw)


def _base(nc, **kw):
    return CLENet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=nc, **kw)


def convnext_small_egb_scg(num_classes: int):
    """CLE-Net-S: ConvNeXt-S + EGB@1-3 + SCG@2-3。"""
    return _small(num_classes, egb_stages=(1, 2, 3), scg_stages=(2, 3))


def convnext_base_egb_scg(num_classes: int):
    """CLE-Net-B: ConvNeXt-B + EGB@1-3 + SCG@2-3。"""
    return _base(num_classes, egb_stages=(1, 2, 3), scg_stages=(2, 3))

def convnext_egb_scg_hsc(num_classes: int):
    """CLE-Net 最终形态: baseline + A + B + C"""
    return _tiny(num_classes, egb_stages=(1, 2, 3), scg_stages=(2, 3), use_hsc=True)


def convnext_egb_scg_fbe(num_classes: int):
    """CLE-Net(EGB+SCG) + FBE 频带均衡: 第三创新点候选。

    FBE 落在 EGB/SCG 均未覆盖的 stage0(96ch, 56x56, 最高分辨率), 与 EGB 构成
    粗(stage1-3 位置级边缘门控)->细(stage0 通道级频带均衡)的完整频域设计语言。
    fbe_w 零初始化 => 初始严格等价 convnext_egb_scg。
    """
    return _tiny(num_classes, egb_stages=(1, 2, 3), scg_stages=(2, 3), fbe_stages=(0,))

def convnext_egb_scg_fbe_all(num_classes: int):
    """消融用: FBE 覆盖全部 stage(0-3)。"""
    return _tiny(num_classes, egb_stages=(1, 2, 3), scg_stages=(2, 3), fbe_stages=(0, 1, 2, 3))

def convnext_egb_scg_msf(num_classes: int):
    """CLE-Net(EGB+SCG) + MSF 多尺度融合头: 第三创新点候选。"""
    return _tiny(num_classes, egb_stages=(1, 2, 3), scg_stages=(2, 3), use_msf=True)

def convnext_egb_scg_hsdg(num_classes: int):
    """CLE-Net(EGB+SCG) + HSD-G: 主分类头保持不变, 亚型聚合经零初始化门注入。

    g=0 时与 convnext_egb_scg 严格等价; 训练中亚型辅助 CE 让亚型层学到病理结构,
    门 g 决定有多少"亚型聚合先验"参与最终决策(可解释为病理先验的贡献量)。
    """
    return _tiny(num_classes, egb_stages=(1, 2, 3), scg_stages=(2, 3), use_hsd_gate=True)

def convnext_egb_scg_hsd(num_classes: int):
    """CLE-Net(EGB+SCG) + HSD 层次化亚型解码头。

    结构: 特征 -> 5 路亚型 logits -> 固定聚合矩阵 A(病理定义) -> 3 类 logits
          + 零初始化残差(原分类头)。训练时额外用亚型 CE 监督亚型层。
    """
    return _tiny(num_classes, egb_stages=(1, 2, 3), scg_stages=(2, 3), use_hsd=True)


# ═══════════════ SR: 频域辅助正则化变体(训练期损失分支, 推理路径不变) ═══════════════

def convnext_egb_scg_sr(num_classes: int):
    """M4: CLE-Net(EGB+SCG) + 频域辅助正则化(SR)。

    结构与 convnext_egb_scg 完全一致, 仅在训练期对预分类头特征增加
    Projection->DCT->频带能量的辅助约束(损失模式由 train_cle_fair.py 的
    --sr-mode 选择: csr 无序判别式[主方案] / spr 单调式[仅对照验证])。
    """
    return _tiny(num_classes, egb_stages=(1, 2, 3), scg_stages=(2, 3), use_sr=True)


def convnext_sr(num_classes: int):
    """互补性消融: baseline + 频域辅助正则化(无 EGB/SCG)。"""
    return _tiny(num_classes, use_sr=True)


def convnext_egb_sr(num_classes: int):
    """互补性消融: EGB + 频域辅助正则化。"""
    return _tiny(num_classes, egb_stages=(1, 2, 3), use_sr=True)


def convnext_scg_sr(num_classes: int):
    """互补性消融: SCG + 频域辅助正则化。"""
    return _tiny(num_classes, scg_stages=(2, 3), use_sr=True)
