import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm for BCHW feature maps."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SpatialChannelProjection(nn.Module):
    """1x1 channel projection followed by 3x3 depthwise spatial aggregation."""

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.spatial = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
            bias=bias,
        )

    def forward(self, x):
        return self.spatial(self.project(x))


SCP = SpatialChannelProjection


class ResidualDilatedDenseBlock(nn.Module):
    """A compact RDDB-style block for local texture and wider spatial context."""

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
    """Channel self-attention with SCP projections, matching the SA-CA core."""

    def __init__(self, dim, num_heads=4, bias=True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = LayerNorm2d(dim)
        self.q_proj = SpatialChannelProjection(dim, dim, bias=bias)
        self.k_proj = SpatialChannelProjection(dim, dim, bias=bias)
        self.v_proj = SpatialChannelProjection(dim, dim, bias=bias)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

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
    """Spatial-channel gated FFN used after channel attention."""

    def __init__(self, dim, expansion=2.0, bias=True):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.norm = LayerNorm2d(dim)
        self.proj_in = SpatialChannelProjection(dim, hidden_dim * 2, bias=bias)
        self.proj_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        value, gate = self.proj_in(x).chunk(2, dim=1)
        x = value * F.gelu(gate)
        return shortcut + self.proj_out(x)


class SACATransformerLayer(nn.Module):
    def __init__(self, dim, num_heads=4, expansion=2.0, bias=True):
        super().__init__()
        self.attention = ChannelAttention(dim, num_heads=num_heads, bias=bias)
        self.ffn = GatedFeedForward(dim, expansion=expansion, bias=bias)

    def forward(self, x):
        x = self.attention(x)
        return self.ffn(x)


class SpatialAwareChannelAttention(nn.Module):
    """Spatial-Aware Channel Attention module from Freqformer."""

    def __init__(self, dim, num_heads=4, num_layers=1, expansion=2.0, bias=True):
        super().__init__()
        self.spatial_context = ResidualDilatedDenseBlock(dim, bias=bias)
        self.layers = nn.Sequential(
            *[
                SACATransformerLayer(dim, num_heads=num_heads, expansion=expansion, bias=bias)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        x = self.spatial_context(x)
        return self.layers(x)


SACA = SpatialAwareChannelAttention


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(2, 64, 128, 128, device=device)

    model = SpatialAwareChannelAttention(64, num_heads=8, num_layers=1).to(device)
    print(model)
    output_tensor = model(input_tensor)

    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理!\n")
