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
    def __init__(self, dim, use_egb=False, use_scg=False, drop_rate=0.,
                 layer_scale_init_value=1e-6, base_ratio=0.1, k2=19):
        super().__init__(dim, drop_rate, layer_scale_init_value)
        self.use_egb = use_egb
        self.use_scg = use_scg
        if use_egb:
            self.egb = _PFECore(base_ratio)
            self.egb_g = nn.Parameter(torch.zeros(1))           # 标量门: 高 lr 组训练
        if use_scg:
            self.scg = StripBranch(dim, k2)                     # c3 零初始化 => 恒等

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        if self.use_egb:
            x = x + self.egb_g * self.egb(x) * x                # EGB 乘性自调制
        if self.use_scg:
            x = x + self.scg(x)                                 # SCG 加性残差
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

class CLENet(ConvNeXt):
    """ConvNeXt + {A:EGB@stages, B:SCG@stages, C:HSC} 渐进叠加。"""
    def __init__(self, egb_stages=(), scg_stages=(), use_hsc=False,
                 strip_k2=19, pfe_ratio=0.1, **kw):
        super().__init__(**kw)
        dims = [dl[1].out_channels if i > 0 else dl[0].out_channels
                for i, dl in enumerate(self.downsample_layers)]

        for si in sorted(set(egb_stages) | set(scg_stages)):
            news = []
            for blk in self.stages[si]:
                nb = CLENetBlock(dim=dims[si],
                                 use_egb=(si in egb_stages),
                                 use_scg=(si in scg_stages),
                                 base_ratio=pfe_ratio, k2=strip_k2)
                nb.load_state_dict(blk.state_dict(), strict=False)
                news.append(nb)
            self.stages[si] = nn.Sequential(*news)

        self.use_hsc = use_hsc
        if use_hsc:
            self.hsc = HSCHead(dims[-1])

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x.mean([-2, -1]))                          # GAP (B,C)
        if self.use_hsc:
            x = self.hsc(x)
        return x


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

def convnext_egb_scg_hsc(num_classes: int):
    """CLE-Net 最终形态: baseline + A + B + C"""
    return _tiny(num_classes, egb_stages=(1, 2, 3), scg_stages=(2, 3), use_hsc=True)
