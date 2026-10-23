import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightChannelAttention(nn.Module):
    """The original pooling-free LCA gate."""

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.proj(x))


class ProgressiveLCA(nn.Module):
    """
    CV stitching variant: local detail, coarse context, and residual strength are
    calibrated together after multi-scale feature fusion.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)

        self.local_detail = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.coarse_pool = nn.AdaptiveAvgPool2d(1)
        self.coarse_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.pixel_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.out_norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local_detail(x)
        coarse = torch.sigmoid(self.coarse_gate(self.coarse_pool(x)))
        gate = self.pixel_gate(torch.cat([x, local], dim=1))
        y = x * gate * coarse + local * (1.0 - gate)
        y = y + self.residual_scale * x
        return self.out_norm(y)


class ProgressiveLCAFusion(nn.Module):
    """Fuse a high-level feature into a low-level feature, then apply Progressive-LCA."""

    def __init__(self, low_channels: int, high_channels: int, out_channels: int):
        super().__init__()
        self.low_proj = nn.Conv2d(low_channels, out_channels, kernel_size=1, bias=False)
        self.high_proj = nn.Conv2d(high_channels, out_channels, kernel_size=1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.attn = ProgressiveLCA(out_channels)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        high = F.interpolate(high, size=low.shape[-2:], mode="nearest")
        fused = self.fuse(torch.cat([self.low_proj(low), self.high_proj(high)], dim=1))
        return self.attn(fused)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    low_feature = torch.randn(2, 64, 128, 128, device=device)
    high_feature = torch.randn(2, 128, 64, 64, device=device)
    model = ProgressiveLCAFusion(64, 128, 64).to(device)
    output_tensor = model(low_feature, high_feature)

    print(model)
    print("input_tensor_shape  :", low_feature.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号 CV缝合救星 独家整理!\n")
