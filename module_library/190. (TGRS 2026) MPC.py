from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn


class CBR(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int = 1,
        padding: int | tuple[int, int] | None = None,
        groups: int = 1,
    ) -> None:
        if padding is None:
            padding = (
                tuple(k // 2 for k in kernel_size)
                if isinstance(kernel_size, tuple)
                else kernel_size // 2
            )

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


def get_split_channels(channels: int, ratio: float) -> tuple[int, int]:
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")

    primary_channels = int(channels * ratio)
    secondary_channels = channels - primary_channels
    if primary_channels < 1 or secondary_channels < 1:
        raise ValueError(
            f"Invalid split: channels={channels}, ratio={ratio}, "
            f"primary={primary_channels}, secondary={secondary_channels}"
        )
    return primary_channels, secondary_channels


class StripDepthwiseConv(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        mode: Literal["sequential", "parallel_sum"] = "sequential",
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if mode not in {"sequential", "parallel_sum"}:
            raise ValueError(f"Unsupported mode: {mode}")

        padding = kernel_size // 2
        self.mode = mode
        self.dw_vertical = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=channels,
            bias=False,
        )
        self.dw_horizontal = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "sequential":
            return self.dw_horizontal(self.dw_vertical(x))
        return self.dw_vertical(x) + self.dw_horizontal(x)


class MultiScaleExtractionLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        kernel_sizes: Sequence[int] = (7, 11, 21),
        strip_mode: Literal["sequential", "parallel_sum"] = "sequential",
    ) -> None:
        super().__init__()
        self.pre_cbr = CBR(in_channels, branch_channels, kernel_size=1, padding=0)
        self.branches = nn.ModuleList(
            [
                StripDepthwiseConv(branch_channels, kernel_size, strip_mode)
                for kernel_size in kernel_sizes
            ]
        )
        self.out_channels = in_channels + len(kernel_sizes) * branch_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.pre_cbr(x)
        features = [branch(feature) for branch in self.branches]
        return torch.cat([x, *features], dim=1)


class CoordinateAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fuse = CBR(channels, channels, kernel_size=1, padding=0)
        self.conv_w = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        x_w = x.mean(dim=3, keepdim=True)
        x_h = x.mean(dim=2, keepdim=True).transpose(2, 3)

        feature = self.fuse(torch.cat([x_w, x_h], dim=2))
        feature_w, feature_h = torch.split(feature, [height, width], dim=2)
        feature_h = feature_h.transpose(2, 3)

        weight_w = torch.sigmoid(self.conv_w(feature_w))
        weight_h = torch.sigmoid(self.conv_h(feature_h))
        return x * weight_w * weight_h


class PartialChannelModule(nn.Module):
    def __init__(self, channels: int, split_ratio: float = 0.25) -> None:
        super().__init__()
        self.primary_channels, self.secondary_channels = get_split_channels(
            channels, split_ratio
        )
        self.primary_conv = nn.Conv2d(
            self.primary_channels,
            self.primary_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.coordinate_attention = CoordinateAttention(self.secondary_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_primary, x_secondary = torch.split(
            x, [self.primary_channels, self.secondary_channels], dim=1
        )
        x_primary = self.primary_conv(x_primary)
        x_secondary = self.coordinate_attention(x_secondary)
        return torch.cat([x_primary, x_secondary], dim=1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average_map = torch.mean(x, dim=1, keepdim=True)
        maximum_map = torch.amax(x, dim=1, keepdim=True)
        attention = torch.sigmoid(self.conv(torch.cat([maximum_map, average_map], dim=1)))
        return x * attention


class PartialSpatialModule(nn.Module):
    def __init__(self, channels: int, split_ratio: float = 0.25) -> None:
        super().__init__()
        self.primary_channels, self.secondary_channels = get_split_channels(
            channels, split_ratio
        )
        self.primary_conv = nn.Conv2d(
            self.primary_channels, self.primary_channels, kernel_size=1, bias=True
        )
        self.spatial_attention = SpatialAttention(kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_primary, x_secondary = torch.split(
            x, [self.primary_channels, self.secondary_channels], dim=1
        )
        x_primary = self.primary_conv(x_primary)
        x_secondary = self.spatial_attention(x_secondary)
        return torch.cat([x_primary, x_secondary], dim=1)


class MPC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        mel_channels: Optional[int] = None,
        split_ratio: float = 0.25,
        stride: int = 1,
        strip_mode: Literal["sequential", "parallel_sum"] = "sequential",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        mel_channels = mel_channels or out_channels

        self.mel = MultiScaleExtractionLayer(
            in_channels=in_channels,
            branch_channels=mel_channels,
            kernel_sizes=(7, 11, 21),
            strip_mode=strip_mode,
        )
        self.parcm = PartialChannelModule(self.mel.out_channels, split_ratio)
        self.channel_fusion = CBR(self.mel.out_channels, out_channels, 1, padding=0)
        self.parsm = PartialSpatialModule(out_channels, split_ratio)
        self.output_fusion = CBR(
            in_channels + out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.mel(x)
        x = self.parcm(x)
        x = self.channel_fusion(x)
        x = self.parsm(x)
        return self.output_fusion(torch.cat([identity, x], dim=1))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(2, 64, 128, 128, device=device)
    model = MPC(in_channels=64, out_channels=64).to(device)
    output_tensor = model(input_tensor)

    print(model)
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号 CV缝合救星 独家整理!\n")
