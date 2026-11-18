import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SpatialChannelProjection(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.spatial = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            groups=out_channels,
            bias=bias,
        )

    def forward(self, x):
        return self.spatial(self.project(x))


class ResidualDilatedDenseBlock(nn.Module):
    def __init__(self, dim, growth_rate=None, bias=True):
        super().__init__()
        growth_rate = growth_rate or max(dim // 2, 16)
        self.conv1 = nn.Conv2d(dim, growth_rate, 3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(dim + growth_rate, growth_rate, 3, padding=2, dilation=2, bias=bias)
        self.conv3 = nn.Conv2d(dim + growth_rate * 2, dim, 3, padding=3, dilation=3, bias=bias)
        self.act = nn.GELU()
        self.res_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
        return x + self.res_scale * x3


class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = LayerNorm2d(dim)
        self.q_proj = SpatialChannelProjection(dim, dim, bias=bias)
        self.k_proj = SpatialChannelProjection(dim, dim, bias=bias)
        self.v_proj = SpatialChannelProjection(dim, dim, bias=bias)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        b, c, h, w = x.shape
        n = h * w
        q = self.q_proj(x).reshape(b, self.num_heads, self.head_dim, n)
        k = self.k_proj(x).reshape(b, self.num_heads, self.head_dim, n)
        v = self.v_proj(x).reshape(b, self.num_heads, self.head_dim, n)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        out = torch.matmul(attention, v).reshape(b, c, h, w)
        return shortcut + self.out_proj(out)


class GatedFeedForward(nn.Module):
    def __init__(self, dim, expansion=2.0, bias=True):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.norm = LayerNorm2d(dim)
        self.proj_in = SpatialChannelProjection(dim, hidden_dim * 2, bias=bias)
        self.proj_out = nn.Conv2d(hidden_dim, dim, 1, bias=bias)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        value, gate = self.proj_in(x).chunk(2, dim=1)
        return shortcut + self.proj_out(value * F.gelu(gate))


class SpatialAwareChannelAttention(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=1, expansion=2.0, bias=True):
        super().__init__()
        self.spatial_context = ResidualDilatedDenseBlock(dim, bias=bias)
        layers = []
        for _ in range(num_layers):
            layers.append(ChannelAttention(dim, num_heads=num_heads, bias=bias))
            layers.append(GatedFeedForward(dim, expansion=expansion, bias=bias))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(self.spatial_context(x))


class AntiAliasFrequencySplit(nn.Module):
    """Freqformer-style smooth low/high split with a fixed anti-alias kernel."""

    def __init__(self, channels, levels=2):
        super().__init__()
        self.levels = levels
        kernel = torch.tensor(
            [[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]],
            dtype=torch.float32,
        )
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))

    def forward(self, x):
        low = x
        for level in range(1, self.levels + 1):
            dilation = 2 ** level
            padding = dilation
            low = F.conv2d(low, self.kernel, padding=padding, dilation=dilation, groups=x.shape[1])
        high = x - low
        return low, high


class FreqGateSACA(nn.Module):
    """Frequency-gated SA-CA for moire texture and color-shift aware restoration."""

    def __init__(self, dim, num_heads=4, num_layers=1, low_scale=2, freq_levels=2, expansion=2.0):
        super().__init__()
        if low_scale < 1:
            raise ValueError("low_scale must be >= 1")
        self.low_scale = low_scale
        self.freq_split = AntiAliasFrequencySplit(dim, levels=freq_levels)
        self.high_saca = SpatialAwareChannelAttention(dim, num_heads, num_layers, expansion)
        self.low_saca = SpatialAwareChannelAttention(dim, num_heads, num_layers, expansion)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            SpatialChannelProjection(dim, dim),
        )
        self.freq_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        low, high = self.freq_split(x)
        high_feature = self.high_saca(high)

        if self.low_scale > 1 and min(x.shape[-2:]) >= self.low_scale:
            low_small = F.interpolate(low, scale_factor=1 / self.low_scale, mode="bilinear", align_corners=False)
            low_feature = self.low_saca(low_small)
            low_feature = F.interpolate(low_feature, size=x.shape[-2:], mode="bilinear", align_corners=False)
        else:
            low_feature = self.low_saca(low)

        fused = self.fuse(torch.cat([high_feature, low_feature], dim=1))
        gate = self.freq_gate(torch.cat([high.abs(), (low - x).abs()], dim=1))
        return x + self.residual_scale * fused * gate


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(2, 64, 128, 128, device=device)

    model = FreqGateSACA(64, num_heads=8, num_layers=1, low_scale=2).to(device)
    print(model)
    output_tensor = model(input_tensor)

    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理!\n")
