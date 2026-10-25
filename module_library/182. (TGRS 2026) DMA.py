import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DecayPos1d(nn.Module):
    """双向光谱衰减位置先验，距离越远衰减越强。"""

    def __init__(self, num_heads: int, initial_value: float = 2.0, heads_range: float = 4.0):
        super().__init__()
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float32) / num_heads)
        )
        self.register_buffer("decay", decay)
        self.tau = nn.Parameter(torch.tensor(3.0))

    def forward(self, length: int) -> Tensor:
        index = torch.arange(length, device=self.decay.device, dtype=torch.float32)
        distance = (index[:, None] - index[None, :]).abs()
        distance = torch.exp(distance / self.tau.clamp_min(1e-3))
        return distance * self.decay[:, None, None]


def gaussian_spatial_position(height: int, width: int, sigma: float, device: torch.device) -> Tensor:
    """生成以 patch 中心为峰值的二维高斯空间先验。"""
    center_h, center_w = height // 2, width // 2
    y = torch.arange(height, dtype=torch.float32, device=device).view(-1, 1)
    x = torch.arange(width, dtype=torch.float32, device=device).view(1, -1)
    dist_sq = (y - center_h) ** 2 + (x - center_w) ** 2
    return torch.exp(-dist_sq / (2 * sigma**2))


class DualPathMultiheadAttention(nn.Module):
    """
    CP-DMA: Center-Pixel Guided Dual-Path Multihead Attention.

    输入/输出均为 [B, N, C]，其中 N 应为方形 patch 的 token 数。
    空间路径用中心像素作为唯一 query 做 one-to-all 聚合；
    光谱路径用深度可分离卷积和光谱衰减先验建模通道连续性；
    两条路径通过逐元素乘法进行早期非线性融合。
    """

    def __init__(self, dim: int, heads: int = 4, dim_heads: int = 8, dropout: float = 0.1, sigma: float = 1.0):
        super().__init__()
        inner_dim = heads * dim_heads
        self.dim = dim
        self.heads = heads
        self.dim_heads = dim_heads
        self.inner_dim = inner_dim
        self.scale = dim_heads ** -0.5
        self.sigma = sigma

        # 空间路径：中心 token 作为 Q，全 patch token 作为 K/V。
        self.to_q_center = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        # 光谱路径：depthwise + pointwise 生成 Q/K/V，注入局部空间上下文。
        self.q_spectral = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, inner_dim, kernel_size=1, bias=False),
        )
        self.k_spectral = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, inner_dim, kernel_size=1, bias=False),
        )
        self.v_spectral = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, inner_dim, kernel_size=1, bias=False),
        )

        self.spectral_pos = DecayPos1d(heads)
        self.gaussian_weight = nn.Parameter(torch.tensor(0.01))
        self.final_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def _check_square_tokens(self, n: int) -> int:
        side = int(math.sqrt(n))
        if side * side != n:
            raise ValueError(f"Token number N={n} must be a square number for CP-DMA.")
        return side

    def forward(self, x: Tensor, mask=None) -> Tensor:
        if x.ndim != 3:
            raise ValueError("DualPathMultiheadAttention expects a 3D tensor: [B, N, C].")

        b, n, c = x.shape
        if c != self.dim:
            raise ValueError(f"Input channels {c} do not match dim={self.dim}.")

        side = self._check_square_tokens(n)
        center_idx = n // 2
        h = self.heads
        dh = self.dim_heads

        # 1) 中心像素引导的空间注意力：复杂度从 all-to-all 降为 one-to-all。
        center_token = x[:, center_idx : center_idx + 1, :]
        q = self.to_q_center(center_token).view(b, 1, h, dh).transpose(1, 2)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        k = k.view(b, n, h, dh).transpose(1, 2)
        v = v.view(b, n, h, dh).transpose(1, 2)

        spatial_score = torch.matmul(q * self.scale, k.transpose(-1, -2))
        g_pos = gaussian_spatial_position(side, side, self.sigma, x.device)
        g_pos = g_pos.reshape(1, 1, 1, n)
        spatial_attn = (spatial_score + self.gaussian_weight * g_pos).softmax(dim=-1)
        spatial_out = v * spatial_attn.transpose(-1, -2)
        spatial_out = spatial_out.transpose(1, 2).reshape(b, n, self.inner_dim)

        # 2) 光谱注意力：中心光谱查询 + 全局光谱键 + 双向衰减先验。
        x_2d = x.transpose(1, 2).reshape(b, c, side, side).contiguous()
        q_map = self.q_spectral(x_2d)
        k_map = self.k_spectral(x_2d)
        v_map = self.v_spectral(x_2d)

        center_h, center_w = side // 2, side // 2
        q_center = q_map[:, :, center_h, center_w].view(b, h, dh, 1)
        k_global = k_map.mean(dim=(2, 3)).view(b, h, dh, 1)
        spectral_score = torch.matmul(q_center, k_global.transpose(-1, -2))
        spectral_score = spectral_score + self.spectral_pos(dh).unsqueeze(0)
        spectral_attn = spectral_score.softmax(dim=-1)

        v_spectral = v_map.view(b, h, dh, n)
        spectral_out = torch.matmul(spectral_attn, v_spectral)
        spectral_out = spectral_out.permute(0, 3, 1, 2).reshape(b, n, self.inner_dim)

        # 3) 逐元素乘法融合，让空间与光谱路径互相门控。
        fused = spatial_out * spectral_out
        return self.final_out(fused)


DMA = DualPathMultiheadAttention


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟高光谱 patch token 输入 [B, N, C]，其中 N = H * W。
    input_tensor = torch.randn(2, 1024, 32).to(device)

    # 直接实例化真正的核心创新模块：DualPathMultiheadAttention
    model = DualPathMultiheadAttention(dim=32, heads=4, dim_heads=8, dropout=0.1).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号：CV缝合救星, 独家整理! \n")
