import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


__all__ = ("DFD_C2f",)


# =========================================================
# Direction-aware Edge Enhancement
# =========================================================

class DirectionalEdgeEnhancement(nn.Module):
    """
    Direction-aware edge enhancement.

    Extract horizontal and vertical structural information
    for directional defect patterns.
    """

    def __init__(self, channels):
        super().__init__()

        hidden = max(channels // 2, 8)

        self.reduce = Conv(
            channels,
            hidden,
            1,
            1
        )

        # horizontal directional feature
        self.dw_h = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(1, 5),
            stride=1,
            padding=(0, 2),
            groups=hidden,
            bias=False
        )

        # vertical directional feature
        self.dw_v = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0),
            groups=hidden,
            bias=False
        )

        self.project = Conv(
            hidden * 2,
            channels,
            1,
            1
        )


    def forward(self, x):

        x = self.reduce(x)

        h = self.dw_h(x)
        v = self.dw_v(x)

        out = torch.cat(
            [h, v],
            dim=1
        )

        return self.project(out)



# =========================================================
# High Frequency Enhancement
# =========================================================

class HighFrequencyEnhancement(nn.Module):
    """
    Lightweight high-frequency residual extraction.

    Extracts defect details by removing local smooth response.
    """

    def __init__(self, channels):
        super().__init__()

        self.conv = Conv(
            channels,
            channels,
            3,
            1
        )


    def forward(self, x):

        smooth = F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1
        )

        high = x - smooth

        return self.conv(high)



# =========================================================
# Adaptive Detail Fusion
# =========================================================

class DetailFusion(nn.Module):
    """
    Adaptive fusion between directional and high-frequency features.
    """

    def __init__(self, channels):
        super().__init__()

        self.gate = nn.Sequential(

            nn.Conv2d(
                channels * 2,
                channels,
                1,
                bias=False
            ),

            nn.BatchNorm2d(channels),

            nn.SiLU(),

            nn.Conv2d(
                channels,
                channels,
                1
            ),

            nn.Sigmoid()
        )


    def forward(self, edge, high):

        weight = self.gate(
            torch.cat(
                [edge, high],
                dim=1
            )
        )

        return (
            edge * weight +
            high * (1 - weight)
        )



# =========================================================
# DFD Bottleneck
# =========================================================

class DFD_Bottleneck(nn.Module):
    """
    Bottleneck block used in DFD-C2f.
    """

    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        g=1,
        e=0.5
    ):
        super().__init__()

        hidden = max(int(c2 * e), 1)

        self.cv1 = Conv(
            c1,
            hidden,
            3,
            1
        )

        self.cv2 = Conv(
            hidden,
            c2,
            3,
            1,
            g=g
        )

        self.add = shortcut and c1 == c2


    def forward(self, x):

        y = self.cv2(
            self.cv1(x)
        )

        return x + y if self.add else y



# =========================================================
# DFD-C2f
# =========================================================

class DFD_C2f(nn.Module):
    """
    Direction-aware Frequency Detail C2f.

    Integrates:
        1. Direction-aware edge enhancement
        2. High-frequency defect detail extraction
        3. Adaptive feature fusion

    Compatible with YOLOv8 C2f interface.

    Args:
        c1: input channels
        c2: output channels
        n: number of bottlenecks
        shortcut: shortcut connection
        g: group convolution
        e: expansion ratio
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        g=1,
        e=0.5
    ):
        super().__init__()

        self.c = max(
            int(c2 * e),
            1
        )


        # YOLOv8 C2f first convolution

        self.cv1 = Conv(
            c1,
            2 * self.c,
            1,
            1
        )


        # detail enhancement branches

        self.edge = DirectionalEdgeEnhancement(
            self.c
        )

        self.high = HighFrequencyEnhancement(
            self.c
        )


        self.fusion = DetailFusion(
            self.c
        )


        # bottleneck sequence

        self.m = nn.ModuleList(

            DFD_Bottleneck(
                self.c,
                self.c,
                shortcut,
                g,
                e=1.0
            )

            for _ in range(n)

        )


        # final projection

        self.cv2 = Conv(
            (2 + n) * self.c,
            c2,
            1,
            1
        )



    def forward(self, x):

        # split feature paths

        y = list(
            self.cv1(x).chunk(2, 1)
        )


        # enhance transformed branch

        detail = y[-1]


        edge_feature = self.edge(
            detail
        )

        high_feature = self.high(
            detail
        )


        detail = self.fusion(
            edge_feature,
            high_feature
        )


        y[-1] = detail


        # CSP bottleneck

        y.extend(
            m(y[-1])
            for m in self.m
        )


        return self.cv2(
            torch.cat(
                y,
                dim=1
            )
        )