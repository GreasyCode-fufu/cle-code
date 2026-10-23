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
            nn.SiLU(inplace=True),
        )


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class MultiScaleSEBranch(nn.Module):
    def __init__(self, channels: int, scale: int, reduction: int) -> None:
        super().__init__()
        self.scale = scale
        self.local = nn.Sequential(
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
        x = self.se(self.local(x))
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class EdgeGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        dx = F.pad(torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]), (1, 0, 0, 0))
        dy = F.pad(torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]), (0, 0, 1, 0))
        edge = dx + dy
        return self.proj(torch.cat([x, edge], dim=1))


class ChangeGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.change_proj = ConvBNAct(channels, channels, kernel_size=3, groups=channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, change_hint: Optional[torch.Tensor]) -> torch.Tensor:
        if change_hint is None:
            smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            change_hint = x - smooth
        elif change_hint.shape[-2:] != x.shape[-2:]:
            change_hint = F.interpolate(
                change_hint, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        if change_hint.shape[1] != x.shape[1]:
            change_hint = change_hint.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1, -1)
        change = self.change_proj(change_hint)
        return self.gate(torch.cat([x, change], dim=1))


class ChangeEdgeSTAM(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        scales: Sequence[int] = (1, 2, 4, 8),
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if len(scales) != 4:
            raise ValueError("scales must contain four values")
        if channels % len(scales) != 0:
            raise ValueError(f"channels must be divisible by {len(scales)}, got {channels}")

        branch_channels = channels // len(scales)
        self.norm = nn.GroupNorm(1, channels)
        self.branches = nn.ModuleList(
            [MultiScaleSEBranch(branch_channels, scale, reduction) for scale in scales]
        )
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.edge_gate = EdgeGate(channels)
        self.change_gate = ChangeGate(channels)
        self.out_proj = ConvBNAct(channels, channels, kernel_size=1, padding=0)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(
        self,
        x: torch.Tensor,
        change_hint: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_size = x.shape[-2:]
        chunks = torch.chunk(self.norm(x), len(self.branches), dim=1)
        multiscale = torch.cat(
            [branch(chunk, output_size) for branch, chunk in zip(self.branches, chunks)],
            dim=1,
        )
        scale_gate = self.scale_fusion(multiscale)
        edge_gate = self.edge_gate(x)
        change_gate = self.change_gate(x, change_hint)
        modulated = x * scale_gate * edge_gate * change_gate
        return x + self.residual_scale * self.out_proj(modulated)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(2, 64, 128, 128, device=device)
    model = ChangeEdgeSTAM(64, reduction=16, scales=(1, 2, 4, 8)).to(device)
    output_tensor = model(input_tensor)

    print(model)
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号 CV缝合救星 独家整理!\n")
