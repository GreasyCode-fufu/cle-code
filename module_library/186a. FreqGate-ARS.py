import torch
import torch.nn as nn
import torch.nn.functional as F


class DAConv(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.num_neighbors = kernel_size * kernel_size - 1
        self.weight = nn.Parameter(torch.zeros(channels, self.num_neighbors))

        mask = torch.ones(kernel_size * kernel_size, dtype=torch.bool)
        mask[kernel_size * kernel_size // 2] = False
        self.register_buffer("mask", mask)

    def forward(self, x):
        b, c, h, w = x.shape
        x_pad = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="reflect")
        patches = F.unfold(x_pad, kernel_size=self.kernel_size)
        patches = patches.view(b, c, self.kernel_size * self.kernel_size, h * w)
        neighbors = patches[:, :, self.mask, :]
        weight = torch.softmax(self.weight, dim=-1).view(1, c, self.num_neighbors, 1)
        surround = (neighbors * weight).sum(dim=2).view(b, c, h, w)
        return x - surround


class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, bins=(1, 2, 3, 7)):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.GELU(),
                )
                for bin_size in bins
            ]
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        return torch.cat(
            [
                F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=True)
                for stage in self.stages
            ],
            dim=1,
        )


class FrequencyGate(nn.Module):
    """Estimate semantic high-frequency strength with fixed Laplacian and Sobel cues."""

    def __init__(self, channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(channels + 3, max(8, channels // 4), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(8, channels // 4), 1, kernel_size=1),
            nn.Sigmoid(),
        )

        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.t()
        self.register_buffer("laplacian", lap.view(1, 1, 3, 3))
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, x):
        base = x.mean(dim=1, keepdim=True)
        lap = F.conv2d(base, self.laplacian.to(dtype=x.dtype), padding=1).abs()
        gx = F.conv2d(base, self.sobel_x.to(dtype=x.dtype), padding=1).abs()
        gy = F.conv2d(base, self.sobel_y.to(dtype=x.dtype), padding=1).abs()
        freq = torch.cat([lap, gx, gy], dim=1)
        return self.project(torch.cat([x, freq], dim=1))


class FreqAwareAttention(nn.Module):
    def __init__(self, channels, psp_bins=(1, 2, 3, 7), reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.daconv3 = DAConv(channels, kernel_size=3)
        self.daconv5 = DAConv(channels, kernel_size=5)
        self.psp = PSPModule(channels, hidden, bins=psp_bins)
        self.freq_gate = FrequencyGate(channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2 + hidden * len(psp_bins) + 1, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x):
        b, _, h, w = x.shape
        freq_gate = self.freq_gate(x)
        feat = torch.cat([self.daconv3(x), self.daconv5(x), self.psp(x), freq_gate], dim=1)
        logits = self.fuse(feat)
        logits = logits + torch.log(freq_gate.clamp_min(1e-4))
        attn = torch.softmax(logits.flatten(2), dim=-1).view(b, 1, h, w)
        return attn * h * w, freq_gate


class CoordinateMapper(nn.Module):
    def __init__(self, kernel_size=7, sigma=2.0):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.padding = kernel_size // 2

        radius = kernel_size // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        self.register_buffer("gaussian_kernel", kernel.view(1, 1, kernel_size, kernel_size))

    def forward(self, attn):
        b, _, h, w = attn.shape
        y = torch.linspace(0, 1, h, device=attn.device, dtype=attn.dtype).view(1, 1, h, 1)
        x = torch.linspace(0, 1, w, device=attn.device, dtype=attn.dtype).view(1, 1, 1, w)
        y = y.expand(b, 1, h, w)
        x = x.expand(b, 1, h, w)

        kernel = self.gaussian_kernel.to(device=attn.device, dtype=attn.dtype)
        denom = F.conv2d(attn, kernel, padding=self.padding).clamp_min(1e-6)
        u = (F.conv2d(attn * y, kernel, padding=self.padding) / denom).clamp(0, 1)
        v = (F.conv2d(attn * x, kernel, padding=self.padding) / denom).clamp(0, 1)
        u[:, :, 0, :] = 0
        u[:, :, -1, :] = 1
        v[:, :, :, 0] = 0
        v[:, :, :, -1] = 1
        return torch.cat([v * 2 - 1, u * 2 - 1], dim=1).permute(0, 2, 3, 1).contiguous()


class FreqGateARS(nn.Module):
    """Frequency-gated ARS with residual stabilization for boundary-rich features."""

    def __init__(self, channels, psp_bins=(1, 2, 3, 7), coord_kernel_size=7, coord_sigma=2.0):
        super().__init__()
        self.attention = FreqAwareAttention(channels, psp_bins=psp_bins)
        self.mapper = CoordinateMapper(kernel_size=coord_kernel_size, sigma=coord_sigma)
        self.refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        attn, freq_gate = self.attention(x)
        grid = self.mapper(attn)
        sampled = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        high = sampled - F.avg_pool2d(sampled, kernel_size=3, stride=1, padding=1)
        fused = self.refine(torch.cat([sampled, high * freq_gate], dim=1))
        return sampled + self.res_scale.tanh() * fused


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(2, 64, 128, 128, device=device)
    model = FreqGateARS(channels=64, psp_bins=(1, 2, 3, 7), coord_kernel_size=7, coord_sigma=2.0).to(device)
    output_tensor = model(input_tensor)

    print(model)
    print("input_tensor_shape :", input_tensor.shape)
    print("output_tensor_shape:", output_tensor.shape)
    print("哔哩哔哩/微信公众号 CV缝合救星 独家整理!")
