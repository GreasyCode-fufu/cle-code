"""ConvNeXt-CLE V2 —— 结构化(不可旁路)创新点, 2026-10-25 夜

V1(零门控并联注入)的判决性实验(experiments/gate_zero_test.py)表明: 训练后门控范数
~1e-3, 门控置零 test F1 变化 ±0.35pt —— 可旁路的加法分支在 wd=5e-2 下被压向恒等,
表面提升主要来自训练轨迹重掷, 机制上不可归因。V2 因此改为结构化替换, 使每个创新点
在计算图上不可旁路:

  A-v2  WS  Wavelet Stem(替换): Haar DWT(RGB->12ch@/2) -> 3x3 s2 conv -> /4,
        两级温和下采样 + 显式子带保留, 替换 k4s4 一步 stem; stem LN 预训练保留。
  B-v2  WR  Wavelet-Replace Block(替换): stage3/4 每块的 7x7 dwconv 整体替换为
        WTConv(base_conv k7 直接继承预训练 dwconv 权重 + Haar 级联小波路径,
        wavelet_scale 初始 0.1 => 小波贡献内生, 无零门控)。
  C-v2  CH  Concat multi-scale Head(替换): 分类头输入为
        [GAP(s4); proj(GAP? s3)); proj(s2)] 拼接(2304-d) -> 新 Linear(2304,3),
        结构上强制消费跨尺度证据; 原 norm 应用于 s4 池化向量。

预训练兼容: 通过 MODEL_CHECKPOINT_TRANSFORMS 重映射
  stem: downsample_layers.0.0.* -> .conv.*, .0.1.* -> .norm.*  (A-v2)
  block: stages.{2,3}.*.dwconv.{weight,bias} -> .dwconv.base_conv.*  (B-v2)
新增随机初始化段白名单: dwt / wavelet_convs / wavelet_scale / base_scale /
wt_filter / iwt_filter / msf2_p2 / msf2_p3 / msf2_n2 / msf2_n3。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Convnext import ConvNeXt, Block, LayerNorm
from models.optimize.convnext_cle import WTConv2d, _create_haar_filters


class HaarDWT(nn.Module):
    """固定 Haar 分析滤波(无可学习参数), (B,C,H,W)->(B,4C,H/2,W/2)。"""
    def __init__(self, channels: int):
        super().__init__()
        wt_filter, _ = _create_haar_filters(channels)
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, "HaarDWT 需偶数空间尺寸"
        t = F.conv2d(x, self.wt_filter, stride=2, groups=c, padding=(0, 0))
        return t.reshape(b, c * 4, h // 2, w // 2)


class WaveletStem(nn.Module):
    """A-v2: DWT -> 3x3 s2 -> LayerNorm(channels_first, 预训练可加载)。"""
    def __init__(self, dims0: int = 96):
        super().__init__()
        self.dwt = HaarDWT(3)
        self.conv = nn.Conv2d(3 * 4, dims0, kernel_size=3, stride=2, padding=1)
        self.norm = LayerNorm(dims0, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        return self.norm(self.conv(self.dwt(x)))


class WaveletReplaceBlock(Block):
    """B-v2: 7x7 dwconv -> WTConv(base k7 继承预训练 + wavelet k3 级联)。
    其余(dwconv 后的 LN/pw/act/gamma/shortcut)与原 Block 完全一致。
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__(dim, drop_rate, layer_scale_init_value)
        self.dwconv = WTConv2d(dim, kernel_size=7, bias=True, wt_levels=2, wt_kernel=3)


class ConvNeXtV2(ConvNeXt):
    """ConvNeXt + {A-v2:WS, B-v2:WR@stages(2,3), C-v2:CH} 结构化组合。"""
    def __init__(self, use_ws=False, wr_stages=(), use_ch=False, **kw):
        super().__init__(**kw)
        dims0 = self.downsample_layers[0][0].out_channels
        self.dims = [dims0] + [self.downsample_layers[i][1].out_channels for i in (1, 2, 3)]

        if use_ws:
            self.downsample_layers[0] = WaveletStem(dims0)

        for si in wr_stages:
            self.stages[si] = nn.Sequential(
                *[WaveletReplaceBlock(self.dims[si]) for _ in self.stages[si]])

        self.use_ch = use_ch
        if use_ch:
            d2, d3, d4 = self.dims[1], self.dims[2], self.dims[3]
            self.msf2_p2 = nn.Conv2d(d2, d4, 1, bias=False)
            self.msf2_p3 = nn.Conv2d(d3, d4, 1, bias=False)
            self.msf2_n2 = nn.LayerNorm(d4, eps=1e-6)
            self.msf2_n3 = nn.LayerNorm(d4, eps=1e-6)
            self.head = nn.Linear(3 * d4, self.head.out_features)  # 替换为三尺度拼接输入

    def forward_features(self, x):
        feats = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feats.append(x)
        m4 = self.norm(feats[3].mean([-2, -1]))
        if self.use_ch:
            m3 = self.msf2_n3(self.msf2_p3(feats[2]).mean([-2, -1]))
            m2 = self.msf2_n2(self.msf2_p2(feats[1]).mean([-2, -1]))
            return torch.cat([m4, m3, m2], dim=-1)
        return m4

    def forward(self, x):
        return self.head(self.forward_features(x))


_TINY = dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])


def _v2(num_classes, **kw):
    return ConvNeXtV2(num_classes=num_classes, **{**_TINY, **kw})


def convnext_v2a(num_classes: int):    # A-v2 单独
    return _v2(num_classes, use_ws=True)


def convnext_v2b(num_classes: int):    # B-v2 单独
    return _v2(num_classes, wr_stages=(2, 3))


def convnext_v2c(num_classes: int):    # C-v2 单独
    return _v2(num_classes, use_ch=True)


def convnext_v2ab(num_classes: int):   # A+B
    return _v2(num_classes, use_ws=True, wr_stages=(2, 3))


def convnext_v2abc(num_classes: int):  # A+B+C 最终链
    return _v2(num_classes, use_ws=True, wr_stages=(2, 3), use_ch=True)
