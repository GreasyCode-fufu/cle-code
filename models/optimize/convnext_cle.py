"""ConvNeXt-CLE 创新变体 —— 2026-10-25

针对 CLE(共聚焦激光显微内镜) 三分类的 ConvNeXt-T 改进, 三个递进创新点:
  A. WDS  Wavelet Detail Stem     —— 输入端: Haar 小波高低频分支零门控注入,
       补偿 4x4/stride-4 stem 对细胞级高频纹理的破坏(门控后 stem 主干权重仍预训练)。
  B. WTB Wavelet-Transformer Block —— stage3/4 块内: 在预训练 7x7 dwconv 之后并联
       WTConv(ECCV2024, Haar 级联分解, 感受野随层数指数增长)零门控分支,
       为腺体结构的多尺度自相似提供频带分离的大感受野混合。
  C. MSF Multi-Scale Fusion head  —— 分类端: stage2/3 特征 1x1 投影到 stage4 维度,
       空间对齐后零门控注入 stage4, 使 GAP 分类头获得跨尺度证据
       (细胞级细节在浅层 / 腺体结构在深层)。

工程约束(与 train_cle_fair 严格加载器协同):
  - 原始 ConvNeXt 的所有参数路径保持不变(stem/Block 均为子类或外科手术式替换),
    ImageNet-1k 官方 ema 权重可 100% 加载;
  - 新增参数段名: wds / wds_gate / wtc / wtg / msf_p2 / msf_p3 / msf_g2 / msf_g3,
    需加入 train_cle_fair.MODEL_ALLOWED_MISSING_SEGMENTS 白名单;
  - 全部新分支零初始化门控 => 初始等价于原 ConvNeXt(继承本项目已验证有效的模式)。

WTConv 移植自 module_library/61(ECCV2024), 小波滤波器内联 Haar(db1)常数,
不引入 pywt 依赖(数值与 pywt.Wavelet('db1') 完全一致)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from models.Convnext import ConvNeXt, Block

_S2 = 0.7071067811865476  # 1/sqrt(2)


# ═══════════════ Haar(db1) 小波基础算子 (内联常数, 免 pywt) ═══════════════

def _create_haar_filters(channels: int):
    """与 module_library/61 中 create_wavelet_filter('db1', ...) 数值一致的滤波器组。

    返回 (dec_filters, rec_filters), 各形状 (4*C, 1, 2, 2), 顺序 [LL, LH, HL, HH]。
    对称的 db1 使 dec 与 rec 滤波器相同; conv(stride2) + conv_transpose(stride2)
    恰为分析/合成伴随算子对, 满足完全重构。
    """
    lo = torch.tensor([_S2, _S2])
    hi = torch.tensor([_S2, -_S2])
    outer = lambda a, b: a.unsqueeze(0) * b.unsqueeze(1)  # [i,j] = b[i]*a[j]
    filters = torch.stack([outer(lo, lo), outer(lo, hi), outer(hi, lo), outer(hi, hi)], dim=0)
    filters = filters[:, None].repeat(channels, 1, 1, 1)  # (4C, 1, 2, 2)
    return filters.clone(), filters.clone()


def _wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    return x.reshape(b, c, 4, h // 2, w // 2)


def _inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    return F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConv2d(nn.Module):
    """WTConv (ECCV 2024) 忠实移植, wt_type 固定 db1(Haar)。输入输出 (B,C,H,W) 同形。"""

    def __init__(self, in_channels, kernel_size=5, bias=True, wt_levels=2):
        super().__init__()
        assert wt_levels >= 1
        self.in_channels = in_channels
        self.wt_levels = wt_levels

        wt_filter, iwt_filter = _create_haar_filters(in_channels)
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)
        self.wt_function = partial(_wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(_inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same',
                                   stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same',
                      stride=1, dilation=1, groups=in_channels * 4, bias=False)
            for _ in range(self.wt_levels)])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
            for _ in range(self.wt_levels)])

    def forward(self, x):
        x_ll_in_levels, x_h_in_levels, shapes_in_levels = [], [], []
        curr_x_ll = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))
            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x = self.base_scale(self.base_conv(x)) + next_x_ll
        return x


# ═══════════════ 创新点 A: Wavelet Detail Stem 分支 ═══════════════

class WaveletDetailStem(nn.Module):
    """RGB -> Haar DWT(LL/LH/HL/HH x3通道=12ch @ /2) -> 3x3 s2 conv -> dims0 @ /4。

    显式保留高频子带后投影, 与预训练 stem conv 输出按通道零门控相加。
    """
    def __init__(self, dims0: int = 96):
        super().__init__()
        wt_filter, _ = _create_haar_filters(3)
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
        self.conv = nn.Conv2d(3 * 4, dims0, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x, self.wt_filter, stride=2, groups=c,
                     padding=(0, 0))  # Haar 核 2x2: pad = k//2-1 = 0
        x = x.reshape(b, c * 4, h // 2, w // 2)
        return self.conv(x)


# ═══════════════ 创新点 B: Wavelet-Transformer Block ═══════════════

class WTEBlock(Block):
    """ConvNeXt Block + 零门控 WTConv 并联分支(插在预训练 7x7 dwconv 之后)。"""
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6,
                 wt_kernel=5, wt_levels=2):
        super().__init__(dim, drop_rate, layer_scale_init_value)
        self.wtc = WTConv2d(dim, kernel_size=wt_kernel, bias=False, wt_levels=wt_levels)
        self.wtg = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x + self.wtg.view(1, -1, 1, 1) * self.wtc(x)
        x = x.permute(0, 2, 3, 1)  # (N,C,H,W)->(N,H,W,C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


# ═══════════════ 组合网络 ═══════════════

class ConvNeXtCLE(ConvNeXt):
    """ConvNeXt + {A:WDS, B:WTB@stages, C:MSF, D:WDD} 可组合。

    构造后对 stages 做外科手术式替换(参数名不变), 故官方预训练权重可整体加载。
    D: Wavelet downsample Dual-branch —— 对 baseline 自身 3 个中间下采样层
    (k2s2 conv) 并联 Haar 小波抗混叠分支(DWT 无损 /2 + 1x1 投影), 零门控融合。
    针对 CLE 图像下采样过程中腺体边缘高频逐渐衰减的问题(PFESA/MICCAI2026 动机)。
    """
    def __init__(self, use_wds=False, wtb_stages=(), wt_kernel=5, wt_levels=2,
                 use_msf=False, use_wdd=False, **convnext_kwargs):
        super().__init__(**convnext_kwargs)
        self._dims = [dl[1].out_channels if i > 0 else dl[0].out_channels
                      for i, dl in enumerate(self.downsample_layers)]

        # A: 小波细节 stem 分支
        self.use_wds = use_wds
        if use_wds:
            self.wds = WaveletDetailStem(self._dims[0])
            self.wds_gate = nn.Parameter(torch.zeros(self._dims[0]))

        # B: stage3/4 块升级为 WTEBlock(逐块新建并继承原参数, 参数路径不变)
        for si in wtb_stages:
            new_blocks = []
            for blk in self.stages[si]:
                nb = WTEBlock(dim=self._dims[si],
                              wt_kernel=wt_kernel, wt_levels=wt_levels)
                nb.load_state_dict(blk.state_dict(), strict=False)  # 原 Block 参数平移
                new_blocks.append(nb)
            self.stages[si] = nn.Sequential(*new_blocks)

        # C: 多尺度融合头
        self.use_msf = use_msf
        if use_msf:
            d2, d3, d4 = self._dims[1], self._dims[2], self._dims[3]
            self.msf_p2 = nn.Conv2d(d2, d4, 1, bias=False)
            self.msf_p3 = nn.Conv2d(d3, d4, 1, bias=False)
            self.msf_g2 = nn.Parameter(torch.zeros(d4))
            self.msf_g3 = nn.Parameter(torch.zeros(d4))

        # D: 小波抗混叠下采样对偶分支(3 个中间 downsample 各一条)
        self.use_wdd = use_wdd
        if use_wdd:
            self.wdd = nn.ModuleList()
            self.wdd_gate = nn.ParameterList()
            for i in range(3):
                d_in, d_out = self._dims[i], self._dims[i + 1]
                wt_filter, _ = _create_haar_filters(d_in)
                m = nn.Module()
                m.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
                m.proj = nn.Conv2d(d_in * 4, d_out, 1, bias=False)
                self.wdd.append(m)
                self.wdd_gate.append(nn.Parameter(torch.zeros(d_out)))

    def _downsample(self, i, x):
        """i=1..3 对应 downsample_layers[i], wdd 列表按 0..2 索引。"""
        y = self.downsample_layers[i](x)
        if self.use_wdd:
            j = i - 1
            b, c, h, w = x.shape
            t = F.conv2d(x, self.wdd[j].wt_filter, stride=2, groups=c,
                         padding=(0, 0))
            t = t.reshape(b, c * 4, h // 2, w // 2)
            y = y + self.wdd_gate[j].view(1, -1, 1, 1) * self.wdd[j].proj(t)
        return y

    def forward_features(self, x):
        if self.use_wds:
            extra = self.wds(x)                                   # (B,d0,H/4,W/4)
            x = self.downsample_layers[0][0](x)                   # 预训练 stem conv
            x = x + self.wds_gate.view(1, -1, 1, 1) * extra
            x = self.downsample_layers[0][1](x)                   # 预训练 stem LN
            x = self.stages[0](x)
            feats = [x]
            for i in range(1, 4):
                x = self._downsample(i, x)
                x = self.stages[i](x)
                feats.append(x)
        else:
            feats = []
            for i in range(4):
                x = self._downsample(i, x) if i > 0 else self.downsample_layers[i](x)
                x = self.stages[i](x)
                feats.append(x)

        z = feats[3]
        if self.use_msf:
            hw = z.shape[-2:]
            z = z + self.msf_g3.view(1, -1, 1, 1) * self.msf_p3(
                F.interpolate(feats[2], size=hw, mode='bilinear', align_corners=False))
            z = z + self.msf_g2.view(1, -1, 1, 1) * self.msf_p2(
                F.interpolate(feats[1], size=hw, mode='bilinear', align_corners=False))
        return self.norm(z.mean([-2, -1]))


# ═══════════════ 工厂(注册名即消融链) ═══════════════

_TINY = dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])


def _tiny(num_classes, **kw):
    return ConvNeXtCLE(num_classes=num_classes, **{**_TINY, **kw})


def convnext_wds(num_classes: int):            # A
    return _tiny(num_classes, use_wds=True)


def convnext_wtb(num_classes: int):            # B
    return _tiny(num_classes, wtb_stages=(2, 3))


def convnext_msf(num_classes: int):            # C
    return _tiny(num_classes, use_msf=True)


def convnext_wds_wtb(num_classes: int):        # A+B
    return _tiny(num_classes, use_wds=True, wtb_stages=(2, 3))


def convnext_wds_wtb_msf(num_classes: int):    # A+B+C (最终网络)
    return _tiny(num_classes, use_wds=True, wtb_stages=(2, 3), use_msf=True)


def convnext_wds_wtb_l1(num_classes: int):     # A+B 消融: wt_levels=1
    return _tiny(num_classes, use_wds=True, wtb_stages=(2, 3), wt_levels=1)


def convnext_wds_wtb_s234(num_classes: int):   # A+B 消融: WTB 三个 stage
    return _tiny(num_classes, use_wds=True, wtb_stages=(1, 2, 3))


def convnext_wds_wtb_k3(num_classes: int):     # A+B 消融: 小核
    return _tiny(num_classes, use_wds=True, wtb_stages=(2, 3), wt_kernel=3)


def convnext_wtb_k3(num_classes: int):         # B 单独(k3), 消融表用
    return _tiny(num_classes, wtb_stages=(2, 3), wt_kernel=3)


def convnext_wds_wtb_k3_msf(num_classes: int):  # A+B(k3)+C 最终链
    return _tiny(num_classes, use_wds=True, wtb_stages=(2, 3), wt_kernel=3, use_msf=True)


def convnext_wdd(num_classes: int):             # D 单独: 小波抗混叠下采样对偶分支
    return _tiny(num_classes, use_wdd=True)


def convnext_wds_msf(num_classes: int):         # A+C 两创新点链
    return _tiny(num_classes, use_wds=True, use_msf=True)


def convnext_wtb_msf(num_classes: int):         # B+C 两创新点链
    return _tiny(num_classes, wtb_stages=(2, 3), use_msf=True)


def convnext_wds_wdd_msf(num_classes: int):     # A+D+C 两位置正交链(弃B)
    return _tiny(num_classes, use_wds=True, use_wdd=True, use_msf=True)
