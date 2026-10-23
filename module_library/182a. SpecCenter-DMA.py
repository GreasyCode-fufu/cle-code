import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpectralDecayPrior(nn.Module):
    """可学习光谱距离衰减先验，用于约束通道间连续性。"""

    def __init__(self, heads: int, initial_value: float = 2.0, heads_range: float = 4.0):
        super().__init__()
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(heads, dtype=torch.float32) / heads)
        )
        self.register_buffer("decay", decay)
        self.tau = nn.Parameter(torch.tensor(3.0))

    def forward(self, length: int) -> Tensor:
        index = torch.arange(length, dtype=torch.float32, device=self.decay.device)
        distance = (index[:, None] - index[None, :]).abs()
        distance = torch.exp(distance / self.tau.clamp_min(1e-3))
        return distance * self.decay[:, None, None]


def center_gaussian(side: int, sigma: float, device: torch.device) -> Tensor:
    center = side // 2
    y = torch.arange(side, dtype=torch.float32, device=device).view(-1, 1)
    x = torch.arange(side, dtype=torch.float32, device=device).view(1, -1)
    return torch.exp(-((y - center) ** 2 + (x - center) ** 2) / (2 * sigma**2))


class SpecCenterDMA(nn.Module):
    """
    SpecCenter-DMA: 光谱可靠性增强的中心像素双路径注意力模块。

    保持原 CP-DMA 的 [B, N, C] 输入输出和中心像素引导机制，在此基础上加入：
    1. 中心-邻域一致性可靠性门控；
    2. 光谱一阶差分门控，强化相邻波段突变/连续性信息；
    3. 自适应空间-光谱融合权重，避免固定乘法在噪声 patch 中过度抑制。
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_heads: int = 8,
        dropout: float = 0.1,
        sigma: float = 1.0,
        reduction: int = 4,
    ):
        super().__init__()
        inner_dim = heads * dim_heads
        hidden = max(dim // reduction, 8)
        self.dim = dim
        self.heads = heads
        self.dim_heads = dim_heads
        self.inner_dim = inner_dim
        self.scale = dim_heads ** -0.5
        self.sigma = sigma

        self.to_q_center = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

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

        self.spectral_prior = SpectralDecayPrior(heads)
        self.gaussian_weight = nn.Parameter(torch.tensor(0.01))

        # 中心像素与邻域均值的差异越大，说明 patch 可能跨类别或边界复杂，需要更谨慎地融合。
        self.center_reliability = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, inner_dim),
            nn.Sigmoid(),
        )

        # 光谱差分门控：显式感知相邻 band/通道之间的连续和突变。
        self.spectral_diff_gate = nn.Sequential(
            nn.Conv1d(inner_dim, inner_dim, kernel_size=3, padding=1, groups=inner_dim, bias=False),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # 自适应融合：在乘法融合和加法残差之间动态平衡。
        self.fusion_gate = nn.Sequential(
            nn.Linear(inner_dim * 2, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, inner_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(inner_dim)
        self.final_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def _side(self, n: int) -> int:
        side = int(math.sqrt(n))
        if side * side != n:
            raise ValueError(f"Token number N={n} must be a square number for SpecCenter-DMA.")
        return side

    def forward(self, x: Tensor, mask=None) -> Tensor:
        if x.ndim != 3:
            raise ValueError("SpecCenterDMA expects a 3D tensor: [B, N, C].")

        b, n, c = x.shape
        if c != self.dim:
            raise ValueError(f"Input channels {c} do not match dim={self.dim}.")

        side = self._side(n)
        center_idx = n // 2
        h = self.heads
        dh = self.dim_heads

        center_token = x[:, center_idx : center_idx + 1, :]
        q = self.to_q_center(center_token).view(b, 1, h, dh).transpose(1, 2)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        k = k.view(b, n, h, dh).transpose(1, 2)
        v = v.view(b, n, h, dh).transpose(1, 2)

        g_pos = center_gaussian(side, self.sigma, x.device).reshape(1, 1, 1, n)
        spatial_score = torch.matmul(q * self.scale, k.transpose(-1, -2))
        spatial_attn = (spatial_score + self.gaussian_weight * g_pos).softmax(dim=-1)
        spatial_out = v * spatial_attn.transpose(-1, -2)
        spatial_out = spatial_out.transpose(1, 2).reshape(b, n, self.inner_dim)

        x_2d = x.transpose(1, 2).reshape(b, c, side, side).contiguous()
        q_map = self.q_spectral(x_2d)
        k_map = self.k_spectral(x_2d)
        v_map = self.v_spectral(x_2d)
        center = side // 2
        q_center = q_map[:, :, center, center].view(b, h, dh, 1)
        k_global = k_map.mean(dim=(2, 3)).view(b, h, dh, 1)
        spectral_score = torch.matmul(q_center, k_global.transpose(-1, -2))
        spectral_score = spectral_score + self.spectral_prior(dh).unsqueeze(0)
        spectral_attn = spectral_score.softmax(dim=-1)
        spectral_out = torch.matmul(spectral_attn, v_map.view(b, h, dh, n))
        spectral_out = spectral_out.permute(0, 3, 1, 2).reshape(b, n, self.inner_dim)

        # 光谱差分门控在 token 维做平均后估计通道可靠性，再广播回所有 token。
        spectral_context = spectral_out.mean(dim=1).unsqueeze(-1)
        diff_gate = self.spectral_diff_gate(spectral_context).transpose(1, 2)

        center_feat = center_token.squeeze(1)
        neighbor_mean = x.mean(dim=1)
        reliability = self.center_reliability(torch.cat([center_feat, (center_feat - neighbor_mean).abs()], dim=-1))
        reliability = reliability.unsqueeze(1)

        mul_fusion = spatial_out * spectral_out
        add_fusion = 0.5 * (spatial_out + spectral_out)
        gate_input = torch.cat([spatial_out.mean(dim=1), spectral_out.mean(dim=1)], dim=-1)
        adaptive_gate = self.fusion_gate(gate_input).unsqueeze(1)
        fused = adaptive_gate * mul_fusion + (1.0 - adaptive_gate) * add_fusion
        fused = fused * reliability * diff_gate + self.res_scale * add_fusion
        fused = self.out_norm(fused)
        return self.final_out(fused)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟高光谱 patch token 输入 [B, N, C]，其中 N = H * W。
    input_tensor = torch.randn(2, 1024, 32).to(device)

    # 直接实例化真正的核心创新模块：SpecCenterDMA
    model = SpecCenterDMA(dim=32, heads=4, dim_heads=8, dropout=0.1).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号：CV缝合救星, 独家整理! \n")
