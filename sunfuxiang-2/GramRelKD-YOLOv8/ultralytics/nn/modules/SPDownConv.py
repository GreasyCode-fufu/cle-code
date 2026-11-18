from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic Conv
# ============================================================

def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = (
            d * (k - 1) + 1
            if isinstance(k, int)
            else tuple(d * (x - 1) + 1 for x in k)
        )

    if p is None:
        p = (
            k // 2
            if isinstance(k, int)
            else tuple(x // 2 for x in k)
        )

    return p


class Conv(nn.Module):
    """Standard convolution + BN + SiLU."""

    default_act = nn.SiLU()

    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False
        )

        self.bn = nn.BatchNorm2d(c2)

        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        return self.act(
            self.bn(
                self.conv(x)
            )
        )

    def forward_fuse(self, x):
        return self.act(
            self.conv(x)
        )


# ============================================================
# Learnable Structure-Preserving Pooling
# ============================================================

class MixedStructurePool(nn.Module):
    """
    Learnable mixture of average pooling and max pooling.

    AvgPool:
        suppresses local noise and preserves regional statistics.

    MaxPool:
        preserves strong thin-line responses and salient structures.

    A learnable scalar decides the balance:
        Y = alpha * AvgPool(X) + (1-alpha) * MaxPool(X)
    """

    def __init__(self):
        super().__init__()

        # sigmoid(0) = 0.5
        # Initialization gives AvgPool and MaxPool equal importance.
        self.pool_logit = nn.Parameter(
            torch.tensor(0.0)
        )

    def forward(self, x):

        alpha = torch.sigmoid(
            self.pool_logit
        )

        avg_out = F.avg_pool2d(
            x,
            kernel_size=2,
            stride=2,
            ceil_mode=True
        )

        max_out = F.max_pool2d(
            x,
            kernel_size=2,
            stride=2,
            ceil_mode=True
        )

        return (
            alpha * avg_out
            +
            (1.0 - alpha) * max_out
        )


# ============================================================
# Structure-Preserving Downsampling Convolution
# ============================================================

class SPDownConv(nn.Module):
    """
    Structure-Preserving Downsampling Convolution.

    Designed for wall defect detection with:
        - small targets
        - elongated targets
        - crack-like structures
        - large scale variations

    Args:
        c1: input channels
        c2: output channels
        k: directional convolution kernel size
        reduction: adaptive fusion reduction ratio

    Input:
        [B, c1, H, W]

    Output:
        [B, c2, ceil(H/2), ceil(W/2)]
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 5,
        reduction: int = 4
    ):
        super().__init__()

        if k < 3 or k % 2 == 0:
            raise ValueError(
                "Directional kernel size k must be an odd number >= 3."
            )

        self.c1 = c1
        self.c2 = c2


        # ====================================================
        # Branch 1:
        # Standard semantic downsampling
        # ====================================================

        self.semantic_branch = Conv(
            c1,
            c2,
            k=3,
            s=2
        )


        # ====================================================
        # Branch 2:
        # Direction-sensitive structure extraction
        #
        # Horizontal:
        # 1 × k
        #
        # Vertical:
        # k × 1
        #
        # Depthwise convolution keeps cost low.
        # ====================================================

        self.horizontal_conv = Conv(
            c1,
            c1,
            k=(1, k),
            s=1,
            g=c1
        )

        self.vertical_conv = Conv(
            c1,
            c1,
            k=(k, 1),
            s=1,
            g=c1
        )


        # ====================================================
        # Structure-preserving downsampling
        # ====================================================

        self.structure_pool = MixedStructurePool()


        # Project structure branch to c2 channels
        self.structure_project = Conv(
            c1,
            c2,
            k=1,
            s=1
        )


        # ====================================================
        # Adaptive branch fusion
        #
        # [Semantic, Structure]
        #        ↓
        # Global descriptor
        #        ↓
        # Generate 2 × C branch weights
        # ====================================================

        hidden = max(
            8,
            c2 // reduction
        )

        self.branch_gate = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                2 * c2,
                hidden,
                kernel_size=1,
                bias=True
            ),

            nn.SiLU(),

            nn.Conv2d(
                hidden,
                2 * c2,
                kernel_size=1,
                bias=True
            )
        )


        # ====================================================
        # Final normalization
        # ====================================================

        self.out_bn = nn.BatchNorm2d(c2)
        self.out_act = nn.SiLU()


    def forward(self, x):

        # ====================================================
        # 1. Semantic branch
        # ====================================================

        semantic = self.semantic_branch(x)


        # ====================================================
        # 2. Direction-sensitive structure extraction
        # ====================================================

        horizontal = self.horizontal_conv(x)

        vertical = self.vertical_conv(x)

        structure = (
            horizontal
            +
            vertical
        )


        # ====================================================
        # 3. Structure-preserving downsampling
        # ====================================================

        structure = self.structure_pool(
            structure
        )

        structure = self.structure_project(
            structure
        )


        # ====================================================
        # 4. Adaptive fusion
        # ====================================================

        fusion_context = torch.cat(
            [
                semantic,
                structure
            ],
            dim=1
        )

        weights = self.branch_gate(
            fusion_context
        )

        b = weights.shape[0]

        weights = weights.reshape(
            b,
            2,
            self.c2,
            1,
            1
        )

        # Competition between semantic and structure branches
        weights = torch.softmax(
            weights,
            dim=1
        )

        semantic_weight = weights[:, 0]

        structure_weight = weights[:, 1]


        # ====================================================
        # 5. Weighted aggregation
        # ====================================================

        out = (
            semantic_weight * semantic
            +
            structure_weight * structure
        )


        # ====================================================
        # 6. Output
        # ====================================================

        out = self.out_act(
            self.out_bn(out)
        )

        return out


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    x = torch.randn(
        2,
        128,
        80,
        80
    )

    model = SPDownConv(
        c1=128,
        c2=256,
        k=5,
        reduction=4
    )

    y = model(x)

    print(model)

    print(
        "Input shape :",
        x.shape
    )

    print(
        "Output shape:",
        y.shape
    )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        params
    )