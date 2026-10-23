from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.SiLU(inplace=True),
        )


def split_channels(channels: int, ratio: float) -> tuple[int, int]:
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")
    primary = int(channels * ratio)
    secondary = channels - primary
    if primary < 1 or secondary < 1:
        raise ValueError(f"invalid split for channels={channels}, ratio={ratio}")
    return primary, secondary


class StripDepthwiseConv(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        mode: Literal["sequential", "parallel_sum"] = "sequential",
    ) -> None:
        super().__init__()
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
        strip_mode: Literal["sequential", "parallel_sum"] = "sequential",
    ) -> None:
        super().__init__()
        self.pre = CBR(in_channels, branch_channels, kernel_size=1, padding=0)
        self.branches = nn.ModuleList(
            [
                StripDepthwiseConv(branch_channels, kernel, strip_mode)
                for kernel in (7, 11, 21)
            ]
        )
        self.out_channels = in_channels + 3 * branch_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.pre(x)
        return torch.cat([x, *(branch(base) for branch in self.branches)], dim=1)


class CoordinateAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fuse = CBR(channels, channels, kernel_size=1, padding=0)
        self.conv_w = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        x_w = x.mean(dim=3, keepdim=True)
        x_h = x.mean(dim=2, keepdim=True).transpose(2, 3)
        fused = self.fuse(torch.cat([x_w, x_h], dim=2))
        w, h = torch.split(fused, [height, width], dim=2)
        h = h.transpose(2, 3)
        return x * torch.sigmoid(self.conv_w(w)) * torch.sigmoid(self.conv_h(h))


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        desc = torch.cat([x.amax(dim=1, keepdim=True), x.mean(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(desc))


class PartialChannelSpatialBlock(nn.Module):
    def __init__(self, channels: int, split_ratio: float) -> None:
        super().__init__()
        primary, secondary = split_channels(channels, split_ratio)
        self.primary_channels = primary
        self.secondary_channels = secondary
        self.primary_conv = CBR(primary, primary, kernel_size=3)
        self.channel_attn = CoordinateAttention(secondary)
        self.mix = CBR(channels, channels, kernel_size=1, padding=0)
        self.spatial_attn = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p, s = torch.split(x, [self.primary_channels, self.secondary_channels], dim=1)
        x = torch.cat([self.primary_conv(p), self.channel_attn(s)], dim=1)
        x = self.mix(x)
        p, s = torch.split(x, [self.primary_channels, self.secondary_channels], dim=1)
        return torch.cat([p, self.spatial_attn(s)], dim=1)


class SelectiveDirectionalScan(nn.Module):
    """
    A dependency-light Mamba-style scan: four cumulative directional summaries
    provide long-range cloud-band context without a quadratic attention matrix.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre = CBR(channels, channels, kernel_size=3, groups=channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = CBR(channels * 2, channels, kernel_size=1, padding=0)

    @staticmethod
    def _scan(x: torch.Tensor, dim: int) -> torch.Tensor:
        length = x.shape[dim]
        forward = torch.cumsum(x, dim=dim) / torch.arange(
            1, length + 1, device=x.device, dtype=x.dtype
        ).view([1, 1, length, 1] if dim == 2 else [1, 1, 1, length])
        flipped = torch.flip(x, dims=[dim])
        backward = torch.cumsum(flipped, dim=dim) / torch.arange(
            1, length + 1, device=x.device, dtype=x.dtype
        ).view([1, 1, length, 1] if dim == 2 else [1, 1, 1, length])
        backward = torch.flip(backward, dims=[dim])
        return 0.5 * (forward + backward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.pre(x)
        scan_h = self._scan(local, dim=2)
        scan_w = self._scan(local, dim=3)
        context = 0.5 * (scan_h + scan_w)
        context = context * self.gate(local)
        return self.out(torch.cat([x, context], dim=1))


class RadianceBoundaryGate(nn.Module):
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
        return x * self.proj(torch.cat([x, edge], dim=1))


class MambaCloudMPC(nn.Module):
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
        self.mel = MultiScaleExtractionLayer(in_channels, mel_channels, strip_mode)
        self.partial = PartialChannelSpatialBlock(self.mel.out_channels, split_ratio)
        self.channel_fusion = CBR(self.mel.out_channels, out_channels, kernel_size=1, padding=0)
        self.scan = SelectiveDirectionalScan(out_channels)
        self.boundary_gate = RadianceBoundaryGate(out_channels)
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
        x = self.partial(x)
        x = self.channel_fusion(x)
        x = self.scan(x)
        x = self.boundary_gate(x)
        return self.output_fusion(torch.cat([identity, x], dim=1))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(2, 64, 128, 128, device=device)
    model = MambaCloudMPC(in_channels=64, out_channels=64).to(device)
    output_tensor = model(input_tensor)

    print(model)
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号 CV缝合救星 独家整理!\n")
