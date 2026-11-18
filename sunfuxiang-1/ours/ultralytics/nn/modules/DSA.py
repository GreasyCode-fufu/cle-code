import torch
import torch.nn as nn


# ============================================================
# Adaptive Statistic Channel Attention
# ============================================================

class ASFChannelAttention(nn.Module):
    """
    Adaptive Statistic Fusion Channel Attention

    Difference from standard CBAM:
        CBAM:
            MLP(AvgPool) + MLP(MaxPool)

        ASF:
            alpha * AvgPool + (1-alpha) * MaxPool
                       ↓
                      MLP

    alpha is input-dependent and channel-wise.
    """

    def __init__(self, channels, ratio=16):
        super().__init__()

        hidden = max(1, channels // ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # ----------------------------------------------------
        # Channel-wise adaptive Avg/Max selector
        #
        # Input:
        #   [B, 2C, 1, 1]
        #
        # Output:
        #   [B, C, 1, 1]
        #
        # groups=C means each channel independently learns
        # how to combine its Avg and Max statistics.
        # ----------------------------------------------------
        self.stat_selector = nn.Conv2d(
            in_channels=channels * 2,
            out_channels=channels,
            kernel_size=1,
            groups=channels,
            bias=True
        )

        # Only ONE MLP evaluation is required
        self.fc1 = nn.Conv2d(
            channels,
            hidden,
            kernel_size=1,
            bias=False
        )

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(
            hidden,
            channels,
            kernel_size=1,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = self.avg_pool(x)
        max_v = self.max_pool(x)

        # ----------------------------------------------------
        # Dynamic statistic selector
        # ----------------------------------------------------

        descriptor = torch.cat(
            [avg, max_v],
            dim=1
        )

        alpha = torch.sigmoid(
            self.stat_selector(descriptor)
        )

        # Convex statistic fusion
        stat = (
            alpha * avg
            +
            (1.0 - alpha) * max_v
        )

        # ----------------------------------------------------
        # Channel attention
        # ----------------------------------------------------

        att = self.fc1(stat)
        att = self.relu(att)
        att = self.fc2(att)

        return self.sigmoid(att)


# ============================================================
# Adaptive Statistic Spatial Attention
# ============================================================

class ASFSpatialAttention(nn.Module):
    """
    Adaptive Statistic Fusion Spatial Attention.

    Instead of directly concatenating AvgMap and MaxMap
    followed by a 2->1 7x7 convolution,
    first dynamically select their contributions.
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7)

        padding = (
            3 if kernel_size == 7
            else 1
        )

        # Pixel-wise selector for Avg / Max maps
        self.stat_selector = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=1,
            bias=True
        )

        # Only one spatial statistic map is processed
        self.spatial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_map = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_map = torch.amax(
            x,
            dim=1,
            keepdim=True
        )

        descriptor = torch.cat(
            [avg_map, max_map],
            dim=1
        )

        beta = torch.sigmoid(
            self.stat_selector(descriptor)
        )

        spatial_stat = (
            beta * avg_map
            +
            (1.0 - beta) * max_map
        )

        att = self.spatial_conv(
            spatial_stat
        )

        return self.sigmoid(att)


# ============================================================
# ASF-CBAM
# ============================================================

class ASFCBAM(nn.Module):
    """
    Adaptive Statistic Fusion CBAM.

    Main modifications:
    1. Input-dependent Avg/Max fusion.
    2. Channel MLP evaluated only once.
    3. Spatial statistics compressed from two maps to one
       adaptive statistic map before 7x7 modeling.

    Input:
        [B, C, H, W]

    Output:
        [B, C, H, W]
    """

    def __init__(
        self,
        channels,
        ratio=16,
        kernel_size=7
    ):
        super().__init__()

        self.channel_attention = (
            ASFChannelAttention(
                channels,
                ratio
            )
        )

        self.spatial_attention = (
            ASFSpatialAttention(
                kernel_size
            )
        )

    def forward(self, x):

        # Channel attention
        x = (
            self.channel_attention(x)
            * x
        )

        # Spatial attention
        x = (
            self.spatial_attention(x)
            * x
        )

        return x


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):

        super(SpatialAttention, self).__init__()


        assert kernel_size in (3,7)


        padding = 3 if kernel_size == 7 else 1


        self.conv1 = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=padding,
            bias=False
        )


        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )


        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )


        x = torch.cat(
            [
                avg_out,
                max_out
            ],
            dim=1
        )


        x = self.conv1(x)


        return self.sigmoid(x)


class DynamicCBAM(nn.Module):

    def __init__(
        self,
        channels,
        ratio=16,
        kernel_size=7
    ):
        super().__init__()

        self.channel_attention = (
            ASFChannelAttention(
                channels,
                ratio
            )
        )

        # 完全保留原始 CBAM Spatial Attention
        self.spatial_attention = (
            SpatialAttention(
                kernel_size
            )
        )

    def forward(self, x):

        x = (
            self.channel_attention(x)
            * x
        )

        x = (
            self.spatial_attention(x)
            * x
        )

        return x