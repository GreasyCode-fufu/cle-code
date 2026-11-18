from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: Optional[int] = None,
        groups: int = 1,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class ScaleSEBranch(nn.Module):
    def __init__(self, channels: int, scale: int = 1, reduction: int = 16) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")
        self.scale = scale
        self.spatial_mixer = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, padding=0),
        )
        self.se = SEBlock(channels, reduction)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        if self.scale > 1:
            height, width = x.shape[-2:]
            x = F.interpolate(
                x,
                size=(max(height // self.scale, 1), max(width // self.scale, 1)),
                mode="bilinear",
                align_corners=False,
            )

        x = self.spatial_mixer(x)
        x = self.se(x)

        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class STAM(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        scales: Sequence[int] = (1, 2, 4, 8),
        residual: bool = False,
        bounded_attention: bool = True,
    ) -> None:
        super().__init__()
        if len(scales) != 4:
            raise ValueError("scales must contain four values")
        if channels % len(scales) != 0:
            raise ValueError(f"channels must be divisible by {len(scales)}, got {channels}")

        branch_channels = channels // len(scales)
        self.residual = residual
        self.bounded_attention = bounded_attention
        self.norm = nn.GroupNorm(1, channels)
        self.branches = nn.ModuleList(
            [ScaleSEBranch(branch_channels, scale, reduction) for scale in scales]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        output_size = x.shape[-2:]
        chunks = torch.chunk(self.norm(x), len(self.branches), dim=1)
        features = [
            branch(chunk, output_size)
            for branch, chunk in zip(self.branches, chunks)
        ]
        attention = self.fusion(torch.cat(features, dim=1))
        if self.bounded_attention:
            attention = torch.sigmoid(attention)

        out = identity * attention
        if self.residual:
            out = identity + out
        return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(2, 64, 128, 128, device=device)
    model = STAM(64, reduction=16, scales=(1, 2, 4, 8), residual=True).to(device)
    output_tensor = model(input_tensor)

    print(model)
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号 CV缝合救星 独家整理!\n")
