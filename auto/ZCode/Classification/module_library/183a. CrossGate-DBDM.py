import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossGateDBDM(nn.Module):
    """
    CrossGate-DBDM: edge-frequency guided dynamic block downsampling.

    It keeps the DBDM behavior: [B, C, H, W] -> [B, out_channels, H/2, W/2].
    """

    def __init__(self, in_channels, out_channels, max_offset=0.25):
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_offset = max_offset

        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1),
        )
        self.block_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU(inplace=True),
        )
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # 跨域门控：用边缘高频图控制四块采样结果的可信度
        self.cross_gate = nn.Sequential(
            nn.Conv2d(out_channels + 1, max(out_channels // 4, 16), kernel_size=1, bias=False),
            nn.BatchNorm2d(max(out_channels // 4, 16)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(out_channels // 4, 16), out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1))

        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _edge_energy(self, x):
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1).abs()
        gy = F.conv2d(gray, self.sobel_y, padding=1).abs()
        edge = gx + gy
        edge = edge / (edge.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return edge

    def _base_grid(self, height, width, device, dtype):
        h, w = height // 2, width // 2
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device, dtype=dtype),
            torch.linspace(-1, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        base = torch.stack([xx, yy], dim=-1)
        block_offsets = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]],
            device=device,
            dtype=dtype,
        ).view(4, 1, 1, 2)
        return base.unsqueeze(0) * 0.5 + block_offsets

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError("CrossGateDBDM expects even spatial dimensions.")

        h, w = H // 2, W // 2
        edge = self._edge_energy(x)
        offset_input = torch.cat([x, edge], dim=1)
        offset = self.offset_conv(offset_input)
        offset = torch.tanh(offset).view(B, 4, 2, h, w).permute(0, 1, 3, 4, 2)
        offset = offset * self.max_offset

        grid = torch.clamp(self._base_grid(H, W, x.device, x.dtype).unsqueeze(0) + offset, -1, 1)
        blocks = []
        for index in range(4):
            sampled = F.grid_sample(
                x,
                grid[:, index],
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            blocks.append(self.block_conv(sampled))

        fuse_feat = torch.cat(blocks, dim=1)
        residual = self.residual_conv(x)
        edge_down = F.avg_pool2d(edge, kernel_size=2, stride=2)
        gate = self.cross_gate(torch.cat([fuse_feat + residual, edge_down], dim=1))
        out = (fuse_feat * gate) + residual + self.res_scale * (fuse_feat - residual)
        return self.final_conv(out)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟遥感检测骨干特征输入 [B, C, H, W]
    input_tensor = torch.randn(2, 64, 32, 32).to(device)

    # 直接实例化真正的核心创新模块：CrossGateDBDM
    model = CrossGateDBDM(in_channels=64, out_channels=128).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")
