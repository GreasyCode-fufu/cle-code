import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class FreqGateDWConvAttention(nn.Module):
    """
    FreqGate-DWConvAttention: 频率保护门控深度卷积注意力模块。

    原始 DWConvAttention 用 depthwise convolution 替代部分局部 attention head，
    速度友好，但固定局部卷积容易在密集预测中抹掉边缘、高频纹理和小目标响应。
    本模块在保持 [B, N, C] 输入输出不变的前提下，加入：
    1. 多尺度 depthwise 局部聚合；
    2. 高频残差感知门控；
    3. 通道可靠性校准；
    4. 可学习残差强度，便于插入预训练 ViT 后微调。
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
        reduction: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")

        self.dim = dim
        self.input_height = input_height
        self.input_width = input_width if input_width is not None else input_height
        hidden = max(dim // reduction, 8)

        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # 多尺度局部聚合：3x3 保边缘，5x5 提供稍大的局部上下文。
        self.dw3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=qkv_bias)
        self.dw5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=qkv_bias)

        # 高频残差来自 x - avgpool(x)，用于识别边缘/纹理位置，避免卷积近似过平滑。
        self.high_freq_gate = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # 通道可靠性：让更像局部模式的通道吃更多 DWConv，噪声通道保持克制。
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1),
            nn.Sigmoid(),
        )

        self.mix = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

        self.init_local_kernels()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("FreqGateDWConvAttention expects a 3D tensor: [B, N, C].")

        B, N, C = x.shape
        expected_patches = self.input_height * self.input_width
        if N != expected_patches:
            raise ValueError(
                f"Sequence length N={N} must equal input_height * input_width = {expected_patches}."
            )
        if C != self.dim:
            raise ValueError(f"Input channels {C} do not match dim={self.dim}.")

        v = self.v_proj(x)
        v_2d = v.transpose(1, 2).reshape(B, C, self.input_height, self.input_width).contiguous()

        smooth = F.avg_pool2d(v_2d, kernel_size=3, stride=1, padding=1)
        high_freq = v_2d - smooth

        local3 = self.dw3(v_2d)
        local5 = self.dw5(v_2d)
        local = self.mix(torch.cat([local3, local5], dim=1))

        spatial_gate = self.high_freq_gate(torch.cat([local, high_freq], dim=1))
        channel_gate = self.channel_gate(high_freq.abs())
        y_2d = local * spatial_gate * channel_gate + self.res_scale * high_freq

        y = y_2d.flatten(2).transpose(1, 2).contiguous()
        y = self.attn_drop(y)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y

    def init_local_kernels(self) -> None:
        """用平滑核初始化局部分支，降低替换预训练注意力后的初始扰动。"""
        with torch.no_grad():
            for conv in (self.dw3, self.dw5):
                k = conv.kernel_size[0]
                coords = torch.arange(k, dtype=conv.weight.dtype) - k // 2
                yy, xx = torch.meshgrid(coords, coords, indexing="ij")
                sigma = max(k / 3.0, 1.0)
                kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                kernel = kernel / kernel.sum()
                kernel = kernel.view(1, 1, k, k).expand_as(conv.weight).clone()
                conv.weight.copy_(kernel)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟 ViT patch token 输入 [B, N, C]，其中 N = H * W。
    input_tensor = torch.randn(2, 1024, 32).to(device)

    # 直接实例化真正的核心创新模块：FreqGateDWConvAttention
    model = FreqGateDWConvAttention(
        dim=32,
        input_height=32,
        input_width=32,
        num_heads=8,
        qkv_bias=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号：CV缝合救星, 独家整理! \n")
