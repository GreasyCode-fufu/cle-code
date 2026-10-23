import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScaleAwarePartialAttention(nn.Module):
    """
    ScaleAware-PartialAttention: 尺度感知通道选择部分注意力模块。

    原始 Partial Attention 固定选择前 ratio 部分通道执行 MHSA。这里在保持
    [B, N, C] 输入输出不变的基础上加入：
    1. 尺度上下文估计，感知 token 特征是否偏小目标/大目标；
    2. 动态通道门控，对 active 通道做样本级重标定；
    3. 局部 token 平滑分支，补偿 frozen 通道完全旁路时的局部细节不足。
    """

    def __init__(
        self,
        dim: int,
        ratio: float = 0.5,
        num_heads: int = 8,
        qkv_bias: bool = False,
        reduction: int = 4,
        local_kernel: int = 3,
    ):
        super().__init__()
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1].")

        active_dim = int(math.ceil(dim * ratio))
        active_dim = min(dim, int(math.ceil(active_dim / num_heads) * num_heads))
        if active_dim % num_heads != 0:
            raise ValueError("Active channel dimension must be divisible by num_heads.")

        self.dim = dim
        self.active_dim = active_dim
        self.frozen_dim = dim - active_dim
        self.num_heads = num_heads
        self.head_dim = active_dim // num_heads
        self.scale = self.head_dim ** -0.5

        hidden = max(active_dim // reduction, 8)
        self.norm = nn.LayerNorm(active_dim)
        self.qkv = nn.Linear(active_dim, active_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(active_dim, active_dim)

        # 尺度上下文门控：用 token 均值和局部方差判断当前样本更偏细粒度还是大尺度目标。
        self.scale_gate = nn.Sequential(
            nn.Linear(active_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, active_dim),
            nn.Sigmoid(),
        )

        # token 维轻量局部交互，不删除 token，只补充邻域连续性。
        padding = local_kernel // 2
        self.local_mixer = nn.Conv1d(
            active_dim,
            active_dim,
            kernel_size=local_kernel,
            padding=padding,
            groups=active_dim,
            bias=False,
        )

        self.frozen_gate = (
            nn.Sequential(
                nn.LayerNorm(self.frozen_dim),
                nn.Linear(self.frozen_dim, self.frozen_dim),
                nn.Sigmoid(),
            )
            if self.frozen_dim > 0
            else None
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("ScaleAwarePartialAttention expects a 3D tensor: [B, N, C].")

        b, n, c = x.shape
        if c != self.dim:
            raise ValueError(f"Input channels {c} do not match dim={self.dim}.")

        active = x[:, :, : self.active_dim]
        frozen = x[:, :, self.active_dim :]

        norm_active = self.norm(active)
        qkv = self.qkv(norm_active)
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_out = (attn @ v).transpose(1, 2).reshape(b, n, self.active_dim)

        local_out = self.local_mixer(active.transpose(1, 2)).transpose(1, 2)
        token_mean = norm_active.mean(dim=1)
        token_var = norm_active.var(dim=1, unbiased=False)
        gate = self.scale_gate(torch.cat([token_mean, token_var], dim=-1)).unsqueeze(1)
        active_out = active + self.proj(attn_out * gate) + self.res_scale * local_out

        if self.frozen_dim > 0:
            frozen_out = frozen * self.frozen_gate(frozen)
            return torch.cat([active_out, frozen_out], dim=-1)
        return active_out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟 ViT token 输入 [B, N, C]
    input_tensor = torch.randn(2, 256, 128).to(device)

    # 直接实例化真正的核心创新模块：ScaleAwarePartialAttention
    model = ScaleAwarePartialAttention(dim=128, ratio=0.5, num_heads=8).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号：CV缝合救星, 独家整理! \n")
