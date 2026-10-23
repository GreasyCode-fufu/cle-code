import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class DWConvAttention(nn.Module):
    """
    DWConvAttention: 用深度卷积近似部分注意力头的局部空间聚合。

    论文核心思想是：若某些 ViT attention head 学到的是局部、近似平移不变、
    输入变化较小的模式，就可以把 Value 投影后的 token 重新折回二维特征图，
    再用 depthwise convolution 做低成本的空间聚合。

    输入/输出均为 [B, N, C]，其中 N = input_height * input_width。
    """

    def __init__(
        self,
        dim: int,
        input_height: int,
        input_width: Optional[int] = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_conv: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")

        self.input_height = input_height
        self.input_width = input_width if input_width is not None else input_height
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        # Value 投影：保留预训练注意力层中的通道混合能力。
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # 深度卷积：每个通道独立学习一个局部空间核，对应论文中的 DW formulation。
        self.dw_conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
            bias=qkv_bias,
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if init_conv:
            self.init_conv()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("DWConvAttention expects a 3D tensor: [B, N, C].")

        B, N, C = x.shape
        expected_patches = self.input_height * self.input_width
        if N != expected_patches:
            raise ValueError(
                f"Sequence length N={N} must equal input_height * input_width = {expected_patches}. "
                f"Current config: input_height={self.input_height}, input_width={self.input_width}."
            )
        if C != self.dim:
            raise ValueError(f"Input channels {C} do not match dim={self.dim}.")

        v = self.v_proj(x)
        v_2d = v.transpose(1, 2).reshape(B, C, self.input_height, self.input_width).contiguous()
        y = self.dw_conv(v_2d)
        y = y.flatten(2).transpose(1, 2).contiguous()
        y = self.attn_drop(y)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y

    def init_conv(self, size: int = 3, sigma: float = 1.0) -> None:
        """用高斯核初始化 depthwise 卷积，让初始局部聚合更平滑。"""
        coords = torch.arange(size, dtype=self.dw_conv.weight.dtype) - size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, size, size).expand_as(self.dw_conv.weight).clone()
        with torch.no_grad():
            self.dw_conv.weight.copy_(kernel)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟 ViT patch token 输入 [B, N, C]，其中 N = H * W。
    input_tensor = torch.randn(2, 1024, 32).to(device)

    # 直接实例化真正的核心创新模块：DWConvAttention
    model = DWConvAttention(
        dim=32,
        input_height=32,
        input_width=32,
        num_heads=8,
        qkv_bias=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        init_conv=True,
    ).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号：CV缝合救星, 独家整理! \n")
