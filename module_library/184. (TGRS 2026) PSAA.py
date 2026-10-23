import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimAM(nn.Module):
    """无参数神经元能量注意力，用局部能量突出小目标响应。"""

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        n = max(h * w - 1, 1)
        x_minus_mu_square = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        energy = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=(2, 3), keepdim=True) / n + self.e_lambda)
        ) + 0.5
        return self.activation(energy)


class HaarHighFrequency(nn.Module):
    """纯 PyTorch Haar 高频分解，提取水平/垂直/对角细节分量。"""

    def __init__(self, channels: int):
        super().__init__()
        hl = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) * 0.5
        lh = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) * 0.5
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) * 0.5
        filters = torch.stack([hl, lh, hh], dim=0).view(3, 1, 2, 2)
        filters = filters.repeat(channels, 1, 1, 1)
        self.channels = channels
        self.register_buffer("filters", filters)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        high = F.conv2d(x, self.filters, stride=2, groups=c)
        high = high.view(b, c, 3, high.shape[-2], high.shape[-1]).sum(dim=2)
        return F.interpolate(high, size=(h, w), mode="nearest")


class WaveletDynamicEnhancer(nn.Module):
    """WDE 分支：用 Haar 高频响应生成局部细节增强权重。"""

    def __init__(self, dim: int):
        super().__init__()
        self.value_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.query_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.haar_high = HaarHighFrequency(dim)
        self.high_proj = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        value = self.value_dw(x)
        query = self.query_dw(x)
        high = self.haar_high(query)
        weight = self.high_proj(high)
        return value * weight


class ContextAwareAmplifier(nn.Module):
    """CAA 分支：结合低频上下文和 SimAM 能量注意力建模全局响应。"""

    def __init__(self, dim: int = 36, scale: int = 8):
        super().__init__()
        self.down_scale = scale
        self.expand = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.low_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.act = nn.GELU()
        self.local_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.wde = WaveletDynamicEnhancer(dim)
        self.simam = SimAM()
        hidden = max(dim // 4, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        local_part, context_part = self.expand(x).chunk(2, dim=1)
        pooled_h = max(h // self.down_scale, 1)
        pooled_w = max(w // self.down_scale, 1)
        low = F.adaptive_avg_pool2d(context_part, (pooled_h, pooled_w))
        low = self.local_dw(low)
        low = F.interpolate(self.act(self.low_proj(low * self.alpha)), size=(h, w), mode="nearest")
        energy = self.simam(context_part)
        context_weight = low + energy * self.beta
        context_out = context_part * context_weight
        detail_out = self.wde(local_part)
        out = context_out + detail_out
        out = self.channel_gate(out) * out
        return self.out_proj(out)


class PSAA(nn.Module):
    """
    Parallel Self-Awareness Attention.

    输入/输出为 [B, C, H, W]。CAA 分支建模全局上下文，WDE 分支增强
    高频细节，最后通过 3x3 与 1x1 双路径卷积输出。
    """

    def __init__(self, in_dim: int, out_dim: int, scale: int = 8):
        super().__init__()
        self.lg_att = ContextAwareAmplifier(in_dim, scale)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_f = x + self.lg_att(x)
        return self.conv3(x_f) + self.conv1(x_f)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟红外小目标检测特征输入 [B, C, H, W]
    input_tensor = torch.randn(2, 32, 128, 128).to(device)

    # 直接实例化真正的核心创新模块：PSAA
    model = PSAA(32, 32, 8).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号：CV缝合救星, 独家整理! \n")
