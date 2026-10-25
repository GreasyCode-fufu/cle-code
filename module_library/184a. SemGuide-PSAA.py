import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimAM(nn.Module):
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        n = max(h * w - 1, 1)
        delta = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        energy = delta / (4 * (delta.sum(dim=(2, 3), keepdim=True) / n + self.e_lambda)) + 0.5
        return self.activation(energy)


class HaarHighFrequency(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hl = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) * 0.5
        lh = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) * 0.5
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) * 0.5
        filters = torch.stack([hl, lh, hh], dim=0).view(3, 1, 2, 2).repeat(channels, 1, 1, 1)
        self.register_buffer("filters", filters)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
        high = F.conv2d(x, self.filters, stride=2, groups=c)
        high = high.view(b, c, 3, high.shape[-2], high.shape[-1]).sum(dim=2)
        return F.interpolate(high, size=(h, w), mode="nearest")


class SemGuidePSAA(nn.Module):
    """
    SemGuide-PSAA: 语义显著性引导的并行自感知注意力模块。

    在原始 PSAA 的 CAA/WDE 双分支基础上，新增：
    1. 语义显著性分支，估计小目标相对背景的响应区域；
    2. 边缘梯度门控，保护红外小目标边界和弱高频信号；
    3. 动态局部-全局融合，按输入场景调节上下文与细节占比。
    输入/输出保持 [B, C, H, W]。
    """

    def __init__(self, in_dim: int, out_dim: int, scale: int = 8, reduction: int = 4):
        super().__init__()
        hidden = max(in_dim // reduction, 8)
        self.scale = scale
        self.expand = nn.Conv2d(in_dim, in_dim * 2, kernel_size=1)
        self.simam = SimAM()
        self.haar = HaarHighFrequency(in_dim)

        self.context_dw = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim)
        self.context_proj = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.GELU(),
        )
        self.detail_proj = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=1),
            nn.Tanh(),
        )

        # 语义显著性：融合平均/最大响应，给出 target-like 区域先验。
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(in_dim + 2, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, in_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # Sobel 边缘门控，显式保护弱边缘和小目标轮廓。
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.edge_gate = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, in_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim * 2, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, in_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, in_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def _edge_response(self, x: Tensor) -> Tensor:
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        return edge / (edge.mean(dim=(2, 3), keepdim=True) + 1e-6)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("SemGuidePSAA expects a 4D tensor: [B, C, H, W].")

        _, _, h, w = x.shape
        local_part, context_part = self.expand(x).chunk(2, dim=1)

        pooled_h = max(h // self.scale, 1)
        pooled_w = max(w // self.scale, 1)
        context = F.adaptive_avg_pool2d(context_part, (pooled_h, pooled_w))
        context = self.context_dw(context)
        context = F.interpolate(self.context_proj(context), size=(h, w), mode="nearest")
        context = context_part * (context + self.simam(context_part))

        high = self.haar(local_part)
        detail = local_part * self.detail_proj(high)

        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True)[0]
        semantic = self.semantic_gate(torch.cat([x, avg_map, max_map], dim=1))
        edge_gate = self.edge_gate(self._edge_response(x))

        fusion = self.fusion_gate(torch.cat([context, detail], dim=1))
        fused = fusion * context + (1.0 - fusion) * detail
        fused = fused * semantic * edge_gate + self.res_scale * (context + detail)
        fused = self.channel_gate(fused) * fused
        x_f = x + fused
        return self.conv3(x_f) + self.conv1(x_f)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟红外小目标检测特征输入 [B, C, H, W]
    input_tensor = torch.randn(2, 32, 128, 128).to(device)

    # 直接实例化真正的核心创新模块：SemGuidePSAA
    model = SemGuidePSAA(32, 32, 8).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号：CV缝合救星, 独家整理! \n")
