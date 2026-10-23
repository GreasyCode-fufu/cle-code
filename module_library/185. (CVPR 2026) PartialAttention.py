import math

import torch
import torch.nn as nn
from torch import Tensor


class PartialAttention(nn.Module):
    """
    Partial Attention from Partial ViT.

    输入/输出均为 [B, N, C]。模块沿通道维把特征拆成 active/frozen 两部分：
    active 通道执行标准多头自注意力，frozen 通道直接旁路保留，从而在不删 token
    的前提下降低 ViT 的注意力计算成本。
    """

    def __init__(self, dim: int, ratio: float = 0.5, num_heads: int = 8, qkv_bias: bool = False):
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

        self.norm = nn.LayerNorm(active_dim)
        self.qkv = nn.Linear(active_dim, active_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(active_dim, active_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("PartialAttention expects a 3D tensor: [B, N, C].")

        b, n, c = x.shape
        if c != self.dim:
            raise ValueError(f"Input channels {c} do not match dim={self.dim}.")

        active = x[:, :, : self.active_dim]
        frozen = x[:, :, self.active_dim :]

        qkv = self.qkv(self.norm(active))
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        active_attn = (attn @ v).transpose(1, 2).reshape(b, n, self.active_dim)
        active_out = active + self.proj(active_attn)

        return torch.cat([active_out, frozen], dim=-1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟 ViT token 输入 [B, N, C]
    input_tensor = torch.randn(2, 256, 128).to(device)

    # 直接实例化真正的核心创新模块：PartialAttention
    model = PartialAttention(dim=128, ratio=0.5, num_heads=8).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号：CV缝合救星, 独家整理! \n")
