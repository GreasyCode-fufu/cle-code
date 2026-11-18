# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from .modules.nonLocal import NLBlockND
from .modules.ghostnet import C2f_GhostBottleneckV2
from .modules.Focusedlinear import FocusedLinearAttention
from .modules.BSAM import BSAM
# from .modules.TransXnet import OSRAAttention,TransxnetHybridTokenMixer
from .modules.MCA import MCALayer
from .modules.DCNV import DCNV2
from .modules.PConv import PConv
from .modules.RFA import CBAMConv,CAMConv,CAConv,RFAConv,RFCAConv
from .modules.backbone.vanillaNet import VanillaBlock
from .modules.CPCA import CPCA
from .modules.CPCA1 import CPCA1
from .modules.CPCA2 import CPCA2
from .modules.CBAM import CBAM
from .modules.DSA import ASFCBAM


from ultralytics.nn.head.head_improve import Detect_improve
from .modules.BiFPN import BiFPN_Concat2, BiFPN_Concat3
from .modules.fasterNet import FasterNet
from .modules.SE import SEAttention
from .modules.ECA import EfficientChannelAttention
from .modules.ASFF import Detect_ASFF
from .modules.ConvNeXt import CNeB
from .modules.Slimneck import VoVGSCSP, GSConv
from .block.iAFF import iAFF, AFF
from .modules.DCNV3 import DCNV3
from .modules.SEv2 import SaELayer
from .modules.EMA import EMA_attention
from .modules.spatial_moe import SpatialMoEContext


from .modules.RFB import BasicRFB
from .modules.iRMB import iRMB, C2f_iRMB
from .modules.swinTransformer import SwinTransformer

from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                    RTDETRDecoder, Segment, SCConv_yolov8, C2f_SCConv, Bottleneck_SCConv, AKConv, LightAIFI)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import v8ClassificationLoss, v8DetectionLoss, v8PoseLoss, v8SegmentationLoss
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights, intersect_dicts,
                                           make_divisible, model_info, scale_img, time_sync)

try:
    import thop
except ImportError:
    thop = None

######################  SPD-Conv  ####     start ###############################
 
class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
 
    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
 
######################  SPD-Conv  ####     start ###############################

###############################################################################
import torch
import torch.nn as nn


class SPDChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        weight = self.sigmoid(avg_out + max_out)
        return x * weight


class ALS_SPDConv(nn.Module):
    """
    Attention-guided Lightweight Space-to-Depth Convolution.

    Args:
        c1: input channels
        c2: output channels
        reduction: reduction ratio for channel attention
    """

    def __init__(self, c1, c2, reduction=16):
        super().__init__()

        self.attn = SPDChannelAttention(c1 * 4, reduction)

        self.dwconv = nn.Sequential(
            nn.Conv2d(c1 * 4, c1 * 4, kernel_size=3, stride=1, padding=1, groups=c1 * 4, bias=False),
            nn.BatchNorm2d(c1 * 4),
            nn.SiLU()
        )

        self.pwconv = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        # Space-to-depth transformation
        x = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)

        # Sub-pixel channel recalibration
        x = self.attn(x)

        # Lightweight local feature extraction and channel fusion
        x = self.dwconv(x)
        x = self.pwconv(x)

        return x
###############################################################################

class SPDChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        weight = self.sigmoid(avg_out + max_out)
        return x * weight


class AttentionSPDConv(nn.Module):
    def __init__(self, c1, c2, reduction=16):
        super().__init__()
        self.attn = SPDChannelAttention(c1 * 4, reduction)
        self.conv = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        x = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)

        x = self.attn(x)
        x = self.conv(x)
        return x
###############################################################################
import torch
import torch.nn as nn


class SA_SPD(nn.Module):
    """
    Spatial-aware Weighted Space-to-Depth.

    It adaptively weights four sub-pixel branches at each spatial location
    before concatenating them into the channel dimension.
    """

    def __init__(self, c1):
        super().__init__()

        self.weight_gen = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, 4, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        w = self.weight_gen(x)
        w = torch.softmax(w, dim=1)

        x00 = x[..., ::2, ::2] * w[:, 0:1]
        x10 = x[..., 1::2, ::2] * w[:, 1:2]
        x01 = x[..., ::2, 1::2] * w[:, 2:3]
        x11 = x[..., 1::2, 1::2] * w[:, 3:4]

        return torch.cat([x00, x10, x01, x11], dim=1)
###############################################################################
class WeightedSPD(nn.Module):
    """
    Weighted Space-to-Depth.
    对四个子像素分支引入可学习权重。
    输入:  B, C, H, W
    输出:  B, 4C, H/2, W/2
    """

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.branch_weight = nn.Parameter(torch.ones(4))

    def forward(self, x):
        w = torch.softmax(self.branch_weight, dim=0)

        x00 = x[..., ::2, ::2] * w[0]
        x10 = x[..., 1::2, ::2] * w[1]
        x01 = x[..., ::2, 1::2] * w[2]
        x11 = x[..., 1::2, 1::2] * w[3]

        return torch.cat([x00, x10, x01, x11], dim=self.d)

###############################################################################
import torch.nn.functional as F

class EdgeEnhancedSPD(nn.Module):
    """
    Edge-enhanced Space-to-Depth with learnable edge scale.
    """

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        edge_h = torch.abs(x[..., 1:, :] - x[..., :-1, :])
        edge_h = F.pad(edge_h, (0, 0, 0, 1))

        edge_w = torch.abs(x[..., :, 1:] - x[..., :, :-1])
        edge_w = F.pad(edge_w, (0, 1, 0, 0))

        edge = edge_h + edge_w

        x = x + self.alpha * edge

        x00 = x[..., ::2, ::2]
        x10 = x[..., 1::2, ::2]
        x01 = x[..., ::2, 1::2]
        x11 = x[..., 1::2, 1::2]

        return torch.cat([x00, x10, x01, x11], dim=self.d)

###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEnhancedSPD(nn.Module):
    """
    Edge-enhanced Space-to-Depth.

    This module enhances local edge responses before performing
    the space-to-depth transformation.

    Input:
        x: Tensor with shape (B, C, H, W)

    Output:
        Tensor with shape (B, 4C, H/2, W/2)
    """

    def __init__(self, dimension=1, init_alpha=0.1):
        super().__init__()
        self.d = dimension

        # Learnable edge enhancement strength.
        # A small initial value is more stable than directly using x + edge.
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, x):
        # Local vertical difference
        edge_h = torch.abs(x[..., 1:, :] - x[..., :-1, :])
        edge_h = F.pad(edge_h, (0, 0, 0, 1))

        # Local horizontal difference
        edge_w = torch.abs(x[..., :, 1:] - x[..., :, :-1])
        edge_w = F.pad(edge_w, (0, 1, 0, 0))

        # Edge response
        edge = edge_h + edge_w

        # Edge-enhanced feature
        x = x + self.alpha * edge

        # Space-to-depth transformation
        x00 = x[..., ::2, ::2]
        x10 = x[..., 1::2, ::2]
        x01 = x[..., ::2, 1::2]
        x11 = x[..., 1::2, 1::2]

        return torch.cat([x00, x10, x01, x11], dim=self.d)


class SPDChannelAttention(nn.Module):
    """
    Channel attention for space-to-depth features.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        weight = self.sigmoid(avg_out + max_out)
        return x * weight


class AttentionEdgeEnhancedSPDConv(nn.Module):
    """
    Attention-guided Edge-enhanced Space-to-Depth Convolution.

    Pipeline:
        Edge Enhancement → Space-to-Depth → Channel Attention → 1×1 Conv

    Args:
        c1: input channels
        c2: output channels
        reduction: reduction ratio for channel attention
        init_alpha: initial edge enhancement strength
    """

    def __init__(self, c1, c2, reduction=16, init_alpha=0.1):
        super().__init__()

        self.edge_spd = EdgeEnhancedSPD(dimension=1, init_alpha=init_alpha)

        self.attn = SPDChannelAttention(
            channels=c1 * 4,
            reduction=reduction
        )

        self.conv = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.edge_spd(x)
        x = self.attn(x)
        x = self.conv(x)
        return x

###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPDChannelAttention(nn.Module):
    """
    Channel attention for SPD-like features.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        weight = self.sigmoid(avg_out + max_out)
        return x * weight


class PartialAttentionSPDConv(nn.Module):
    """
    Partial Attention-guided Space-to-Depth Convolution.

    Compared with the original AttentionSPDConv, this module only applies
    Space-to-Depth to part of the input channels. The remaining channels are
    processed by lightweight depthwise downsampling. Then the two branches are
    fused using channel attention and 1×1 convolution.

    Args:
        c1: input channels
        c2: output channels
        spd_ratio: ratio of channels using SPD
        reduction: reduction ratio for channel attention

    Input:
        x:  B, C, H, W

    Output:
        y:  B, c2, H/2, W/2
    """

    def __init__(self, c1, c2, spd_ratio=0.5, reduction=16):
        super().__init__()

        assert 0.0 < spd_ratio <= 1.0, "spd_ratio should be in (0, 1]."

        self.c1 = c1
        self.c2 = c2
        self.spd_ratio = spd_ratio

        self.c_spd = max(1, int(c1 * spd_ratio))
        self.c_down = c1 - self.c_spd

        # Lightweight downsampling branch for non-SPD channels
        if self.c_down > 0:
            self.down = nn.Sequential(
                nn.Conv2d(
                    self.c_down,
                    self.c_down,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=self.c_down,
                    bias=False
                ),
                nn.BatchNorm2d(self.c_down),
                nn.SiLU()
            )
        else:
            self.down = None

        # Channels after partial SPD and lightweight downsampling
        fuse_in = self.c_spd * 4 + self.c_down

        self.attn = SPDChannelAttention(fuse_in, reduction)

        self.conv = nn.Sequential(
            nn.Conv2d(fuse_in, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        # Make H and W even to avoid size mismatch in SPD slicing
        _, _, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            x = F.pad(x, (0, w % 2, 0, h % 2))

        if self.c_down > 0:
            x_spd, x_down = torch.split(x, [self.c_spd, self.c_down], dim=1)
        else:
            x_spd = x
            x_down = None

        # Partial Space-to-Depth branch
        x_spd = torch.cat([
            x_spd[..., ::2, ::2],
            x_spd[..., 1::2, ::2],
            x_spd[..., ::2, 1::2],
            x_spd[..., 1::2, 1::2]
        ], dim=1)

        # Lightweight downsampling branch
        if x_down is not None:
            x_down = self.down(x_down)
            x = torch.cat([x_spd, x_down], dim=1)
        else:
            x = x_spd

        # Attention recalibration and channel fusion
        x = self.attn(x)
        x = self.conv(x)

        return x

###############################################################################
import torch
import torch.nn as nn


class OCSPDChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.mlp(self.pool(x))


class OrthogonalCompactSPDConv(nn.Module):
    """
    Orthogonal Compact Space-to-Depth Convolution.

    It replaces direct space-to-depth concatenation with Haar-like
    orthogonal decomposition and compact high-frequency compression.

    Args:
        c1: input channels
        c2: output channels
        high_ratio: compressed high-frequency channel ratio relative to c1
        reduction: channel attention reduction ratio
    """

    def __init__(self, c1, c2, high_ratio=1.0, reduction=16):
        super().__init__()

        self.c1 = c1
        self.c2 = c2
        self.high_c = max(1, int(c1 * high_ratio))

        # Compress three high-frequency components: 3C -> high_c
        self.high_compress = nn.Sequential(
            nn.Conv2d(c1 * 3, self.high_c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.high_c),
            nn.SiLU()
        )

        compact_c = c1 + self.high_c

        self.attn = OCSPDChannelAttention(compact_c, reduction)

        self.fuse = nn.Sequential(
            nn.Conv2d(compact_c, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        

    def forward(self, x):
        # 2x2 polyphase components
        x00 = x[..., ::2, ::2]
        x10 = x[..., 1::2, ::2]
        x01 = x[..., ::2, 1::2]
        x11 = x[..., 1::2, 1::2]

        # Haar-like orthogonal decomposition
        l  = (x00 + x10 + x01 + x11) * 0.5
        hv = (x00 - x10 + x01 - x11) * 0.5
        hh = (x00 + x10 - x01 - x11) * 0.5
        hd = (x00 - x10 - x01 + x11) * 0.5

        # Compact high-frequency representation
        h = torch.cat([hv, hh, hd], dim=1)
        h = self.high_compress(h)

        # Compact SPD representation
        x = torch.cat([l, h], dim=1)

        x = self.attn(x)
        x = self.fuse(x)

        return x
###############################################################################

class EnergyGatedOCSPDConv(nn.Module):
    def __init__(self, c1, c2, high_ratio=1.0, reduction=16):
        super().__init__()

        self.high_c = max(1, int(c1 * high_ratio))

        self.high_compress = nn.Sequential(
            nn.Conv2d(c1 * 3, self.high_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.high_c),
            nn.SiLU()
        )

        # Gate is generated from explicit high-frequency energy: C -> high_c
        self.energy_gate = nn.Sequential(
            nn.Conv2d(c1, self.high_c, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

        compact_c = c1 + self.high_c

        self.attn = OCSPDChannelAttention(compact_c, reduction)

        self.fuse = nn.Sequential(
            nn.Conv2d(compact_c, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            x = F.pad(x, (0, w % 2, 0, h % 2))

        x00 = x[..., ::2, ::2]
        x10 = x[..., 1::2, ::2]
        x01 = x[..., ::2, 1::2]
        x11 = x[..., 1::2, 1::2]

        l = (x00 + x10 + x01 + x11) * 0.5
        hv = (x00 - x10 + x01 - x11) * 0.5
        hh = (x00 + x10 - x01 - x11) * 0.5
        hd = (x00 - x10 - x01 + x11) * 0.5

        h_raw = torch.cat([hv, hh, hd], dim=1)
        h_compact = self.high_compress(h_raw)

        # Explicit high-frequency energy
        energy = torch.abs(hv) + torch.abs(hh) + torch.abs(hd)

        gate = self.energy_gate(energy)
        h_compact = h_compact * gate

        x = torch.cat([l, h_compact], dim=1)

        x = self.attn(x)
        x = self.fuse(x)

        return x

###############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedEnergyGatedOCSPDConv(nn.Module):
    """
    能量门控 Haar 小波 SPD 卷积模块 (Optimized Version)
    改进点：
    1. 修正高频能量图计算，保留方向选择性与空间窗平滑，防止噪声误触发；
    2. 优化计算图，增强高频门控与高频压缩的表达能力。
    """
    def __init__(self, c1, c2, high_ratio=1.0, reduction=16):
        super().__init__()

        self.high_c = max(1, int(c1 * high_ratio))

        # 1. 高频特征压缩分支：将 3*c1 通道的高频解耦特征压缩至 high_c
        self.high_compress = nn.Sequential(
            nn.Conv2d(c1 * 3, self.high_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.high_c),
            nn.SiLU()
        )

        # 2. 改进的能量门控生成网络：
        # 保留方向高频独立性，并引入 3x3 Depthwise 卷积进行空间能量平滑（抑制孤立噪声点）
        self.energy_smooth = nn.Conv2d(c1 * 3, c1 * 3, kernel_size=3, padding=1, groups=c1 * 3, bias=False)
        self.energy_gate = nn.Sequential(
            nn.Conv2d(c1 * 3, self.high_c, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

        compact_c = c1 + self.high_c

        # 3. 跨频域通道注意力机制 (假设已完成正确定义)
        self.attn = OCSPDChannelAttention(compact_c, reduction)

        # 4. 最终特征融合与通道映射
        self.fuse = nn.Sequential(
            nn.Conv2d(compact_c, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            x = F.pad(x, (0, w % 2, 0, h % 2))

        # 2D Haar 小波下采样与分解
        x00 = x[..., ::2, ::2]
        x10 = x[..., 1::2, ::2]
        x01 = x[..., ::2, 1::2]
        x11 = x[..., 1::2, 1::2]

        l  = (x00 + x10 + x01 + x11) * 0.5  # 低频近似 (LL)
        hv = (x00 - x10 + x01 - x11) * 0.5  # 垂直高频 (LH)
        hh = (x00 + x10 - x01 - x11) * 0.5  # 水平高频 (HL)
        hd = (x00 - x10 - x01 + x11) * 0.5  # 对角线高频 (HH)

        # 拼接原始高频细节 (3 * c1 通道)
        h_raw = torch.cat([hv, hh, hd], dim=1)
        
        # 1. 压缩高频特征
        h_compact = self.high_compress(h_raw)

        # 2. 空间平滑的显式高频能量估计
        # 计算高频绝对值能量，并经过 3x3 DWConv 平滑，防止孤立噪声污染能量门
        raw_energy = torch.abs(h_raw)
        smoothed_energy = self.energy_smooth(raw_energy)
        
        # 生成通道与空间自适应门控掩码
        gate = self.energy_gate(smoothed_energy)
        
        # 门控对高频压缩特征实施选择性过滤
        h_compact = h_compact * gate

        # 3. 低频与过滤后的高频重新拼接 (c1 + high_c)
        x_fused = torch.cat([l, h_compact], dim=1)

        # 4. 注意力重新校准与输出投影
        x_fused = self.attn(x_fused)
        out = self.fuse(x_fused)

        return out
###############################################################################

class BaseModel(nn.Module):
    """
    The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family.
    """

    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model on a single scale.
        Wrapper for `_forward_once` method.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize)

    def _predict_once(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the network.
        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if hasattr(m, 'backbone'):
                x = m(x)
                if len(x) != 5: # 0 - 5
                    x.insert(0, None)
                for index, i in enumerate(x):
                    if index in self.save:
                        y.append(i)
                    else:
                        y.append(None)
                x = x[-1] # 最后一个输出传给下一层
            else:
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(f'WARNING ⚠️ {self.__class__.__name__} does not support augmented inference yet. '
                       f'Reverting to single-scale inference instead.')
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input.
        Appends the results to the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, 'bn'):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, 'bn'):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            A model that is a Detect() object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Detect_ASFF, Detect_improve)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights['model'] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights')

    def loss(self, batch, preds=None):
        """
        Compute loss

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        preds = self.forward(batch['img']) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        raise NotImplementedError('compute_loss() needs to be implemented by task heads')


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose, Detect_ASFF, Detect_improve)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLOv5 augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        return v8DetectionLoss(self)


class SegmentationModel(DetectionModel):
    """YOLOv8 segmentation model."""

    def __init__(self, cfg='yolov8n-seg.yaml', ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLOv8 pose model."""

    def __init__(self, cfg='yolov8n-pose.yaml', ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg['kpt_shape']):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg['kpt_shape'] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLOv8 classification model."""

    def __init__(self, cfg='yolov8n-cls.yaml', ch=3, nc=None, verbose=True):
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override YAML value
        elif not nc and not self.yaml.get('nc', None):
            raise ValueError('nc not specified. Must specify nc in model.yaml or function arguments.')
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        name, m = list((model.model if hasattr(model, 'model') else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = types.index(nn.Linear)  # nn.Linear index
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = types.index(nn.Conv2d)  # nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """Compute the classification loss between predictions and true labels."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):

    def __init__(self, cfg='rtdetr-l.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Compute the classification loss between predictions and true labels."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        img = batch['img']
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch['batch_idx']
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            'cls': batch['cls'].to(img.device, dtype=torch.long).view(-1),
            'bboxes': batch['bboxes'].to(device=img.device),
            'batch_idx': batch_idx.to(img.device, dtype=torch.long).view(-1),
            'gt_groups': gt_groups}

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta['dn_num_split'], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion((dec_bboxes, dec_scores),
                              targets,
                              dn_bboxes=dn_bboxes,
                              dn_scores=dn_scores,
                              dn_meta=dn_meta)
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor([loss[k].detach() for k in ['loss_giou', 'loss_class', 'loss_bbox']],
                                                   device=img.device)

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False
            batch (dict): A dict including gt boxes and labels from dataloader.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt = [], []  # outputs
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLOv5 network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.

    Example:
        ```python
        with temporary_modules({'old.module.path': 'new.module.path'}):
            import old.module.path  # this will now import new.module.path
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if not modules:
        modules = {}

    import importlib
    import sys
    try:
        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = importlib.import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised,
    it catches the error, logs a warning message, and attempts to install the missing module via the
    check_requirements() function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.

    Returns:
        (dict): The loaded PyTorch model.
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix='.pt')
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules({
                'ultralytics.yolo.utils': 'ultralytics.utils',
                'ultralytics.yolo.v8': 'ultralytics.models.yolo',
                'ultralytics.yolo.data': 'ultralytics.data'}):  # for legacy 8.0 Classify and Pose models
            return torch.load(file, map_location='cpu'), file  # load

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == 'models':
            raise TypeError(
                emojis(f'ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained '
                       f'with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with '
                       f'YOLOv8 at https://github.com/ultralytics/ultralytics.'
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")) from e
        LOGGER.warning(f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in ultralytics requirements."
                       f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")
        check_requirements(e.name)  # install missing module

        return torch.load(file, map_location='cpu'), file  # load


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""

    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt['train_args']} if 'train_args' in ckpt else None  # combined args
        model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, 'stride'):
            model.stride = torch.tensor([32.])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment, Detect_ASFF, Detect_improve):
            m.inplace = inplace
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[torch.argmax(torch.tensor([m.stride.max() for m in ensemble])).int()].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f'Models differ in class counts {[m.nc for m in ensemble]}'
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get('train_args', {}))}  # combine model and default args, preferring model args
    model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.])

    model = model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment, Detect_ASFF):
            m.inplace = inplace
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    backbone = False

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        t = m
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3,
                 VoVGSCSP, GSConv, SCConv_yolov8, C2f_SCConv, Bottleneck_SCConv, C2f_iRMB, BasicRFB, CNeB, AKConv, VanillaBlock, CBAMConv, CAMConv,
                 CAConv, RFAConv, RFCAConv, PConv, DCNV2, DCNV3, MCALayer, C2f_GhostBottleneckV2, NLBlockND, EMA_attention):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, VoVGSCSP, C3TR, C3Ghost, C3x, RepC3, C2f_iRMB, DCNV2, DCNV3, C2f_GhostBottleneckV2):
                args.insert(2, n)  # number of repeats
                n = 1

        elif m in {iRMB}:
            args = [ch[f], ch[f]]

        elif m is FocusedLinearAttention:
            args = {*args}

        elif m is SaELayer:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, *args[1:]]

        elif m is CBAM:
            c2 = ch[f]
            args = [c2, *args]

        elif m is ASFCBAM:
            c2 = ch[f]
            args = [c2, *args]

            
        elif m is BSAM:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, *args[1:]]

        elif m is space_to_depth:
            c2 = 4 * ch[f]

        elif m is AttentionSPDConv:
            c1, c2 = ch[f], args[0]
            args = [c1, c2, *args[1:]]

        elif m is ALS_SPDConv:
            c1 = ch[f]
            c2 = 4 * c1
            reduction = args[0] if len(args) else 16
            args = [c1, c2, reduction]

        elif m is SA_SPD:
            c2 = 4 * ch[f]
            args = [ch[f]]

        elif m is WeightedSPD:
            c2 = 4 * ch[f]

        elif m is EdgeEnhancedSPD:
            c2 = 4 * ch[f]

        elif m is AttentionEdgeEnhancedSPDConv:
            c1 = ch[f]
            c2 = 4 * c1
            reduction = args[0] if len(args) > 0 else 16
            init_alpha = args[1] if len(args) > 1 else 0.1
            args = [c1, c2, reduction, init_alpha]

        elif m is PartialAttentionSPDConv:
            c1 = ch[f]
            spd_ratio = args[0] if len(args) > 0 else 0.5
            out_ratio = args[1] if len(args) > 1 else 2.0
            reduction = args[2] if len(args) > 2 else 16

            c2 = int(c1 * out_ratio)
            args = [c1, c2, spd_ratio, reduction]

        elif m is OrthogonalCompactSPDConv:
            c1 = ch[f]
            high_ratio = args[0] if len(args) > 0 else 1.0
            out_ratio = args[1] if len(args) > 1 else 2.0
            reduction = args[2] if len(args) > 2 else 16

            c2 = int(c1 * out_ratio)
            args = [c1, c2, high_ratio, reduction]

        elif m in {EnergyGatedOCSPDConv, OptimizedEnergyGatedOCSPDConv}:
            c1 = ch[f]

            # YAML 参数顺序：
            # [high_ratio, out_ratio, reduction]
            high_ratio = args[0] if len(args) > 0 else 1.0
            out_ratio = args[1] if len(args) > 1 else 2.0
            reduction = args[2] if len(args) > 2 else 16

            c2 = int(c1 * out_ratio)

            # 转换为模块构造函数需要的顺序：
            # (c1, c2, high_ratio, reduction)
            args = [c1, c2, high_ratio, reduction]

        elif m in {iAFF, AFF}:
            channels = [ch[x] for x in f]
            c2 = channels[0]       # output of the iAFF module ( output channel is eighter of the channels)
            args = [c2]

        elif m in {SwinTransformer, FasterNet}:
            m = m()
            c2 = m.width_list
            backbone = True

        elif m in [Concat, BiFPN_Concat2, BiFPN_Concat3]:
            c2 = sum(ch[x] for x in f)

        elif m is CPCA:
            c2 = ch[f]
            args = [c2, *args]

        elif m is CPCA1:
            c2 = ch[f]
            args = [c2, *args]

        
        elif m is CPCA2:
            c2 = ch[f]
            args = [c2, *args]
        # elif m in {Bi_FPN}:
        #     length = len([ch[x] for x in f])
        #     args = [length]

        elif m is SpatialMoEContext:
            c2 = ch[f]
            args = [c2, *args]

        elif m in {AIFI, EfficientChannelAttention, SEAttention, LightAIFI}:
            args = [ch[f], *args]

        
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Segment, Pose, Detect_ASFF, Detect_improve):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        else:
            c2 = ch[f]

        if isinstance(c2, list):
            m_ = m
            m_.backbone = True
        else:
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
    
    
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i + 4 if backbone else i, f, t  # attach index, 'from' index, type

        # m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # t = str(m)[8:-2].replace('__main__.', '')  # module type
        # m.np = sum(x.numel() for x in m_.parameters())  # number params
        # m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        # 
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        save.extend(x % (i + 4 if backbone else i) for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        if isinstance(c2, list):
            ch.extend(c2)
            if len(c2) != 5:
                ch.insert(0, 0)
        else:
            ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    path = Path(path)
    if path.stem in (f'yolov{d}{x}6' for x in 'nsmlx' for d in (5, 8)):
        new_stem = re.sub(r'(\d+)([nslmx])6(.+)?$', r'\1\2-p6\3', path.stem)
        LOGGER.warning(f'WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.')
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r'(\d+)([nslmx])(.+)?$', r'\1\3', str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d['scale'] = guess_model_scale(path)
    d['yaml_file'] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale.
    The function uses regular expression matching to find the pattern of the model scale in the YAML file name,
    which is denoted by n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re
        return re.search(r'yolov\d+([nslmx])', Path(model_path).stem).group(1)  # n, s, m, l, or x
    return ''


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg['head'][-1][-2].lower()  # output module name
        if m in ('classify', 'classifier', 'cls', 'fc'):
            return 'classify'
        if m == 'detect':
            return 'detect'
        if m == 'segment':
            return 'segment'
        if m == 'pose':
            return 'pose'
        else:
            return 'detect'


    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in 'model.args', 'model.model.args', 'model.model.model.args':
            with contextlib.suppress(Exception):
                return eval(x)['task']
        for x in 'model.yaml', 'model.model.yaml', 'model.model.model.yaml':
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        for m in model.modules():
            if isinstance(m, (Detect, Detect_ASFF)):
                return 'detect'
            elif isinstance(m, Segment):
                return 'segment'
            elif isinstance(m, Classify):
                return 'classify'
            elif isinstance(m, Pose):
                return 'pose'

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if '-seg' in model.stem or 'segment' in model.parts:
            return 'segment'
        elif '-cls' in model.stem or 'classify' in model.parts:
            return 'classify'
        elif '-pose' in model.stem or 'pose' in model.parts:
            return 'pose'
        elif 'detect' in model.parts:
            return 'detect'

    # Unable to determine task from model
    LOGGER.warning("WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
                   "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', or 'pose'.")
    return 'detect'  # assume detect
