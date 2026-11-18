import torch
import torch.nn as nn
import torch.nn.functional as F


class DBDM(nn.Module):
    """
    Dynamic Block Downsampling Module for remote sensing object detection.

    Input : [B, C, H, W]
    Output: [B, out_channels, H/2, W/2]
    """

    def __init__(self, in_channels, out_channels, max_offset=0.25):
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4 for four block branches.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_offset = max_offset

        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1),
        )

        self.block_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
        )
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def generate_base_grid(self, height, width, device, dtype):
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
            raise ValueError("DBDM expects even spatial dimensions.")

        h, w = H // 2, W // 2
        offset = self.offset_conv(x)
        offset = torch.tanh(offset).view(B, 4, 2, h, w).permute(0, 1, 3, 4, 2)
        offset = offset * self.max_offset

        base_grid = self.generate_base_grid(H, W, x.device, x.dtype).unsqueeze(0)
        grid = torch.clamp(base_grid + offset, -1, 1)

        sampled_blocks = []
        for index in range(4):
            sampled = F.grid_sample(
                x,
                grid[:, index],
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            sampled_blocks.append(self.block_conv(sampled))

        fuse_feat = torch.cat(sampled_blocks, dim=1)
        residual = self.residual_conv(x)
        return self.final_conv(fuse_feat + residual)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟遥感检测骨干特征输入 [B, C, H, W]
    input_tensor = torch.randn(2, 64, 32, 32).to(device)

    # 直接实例化真正的核心创新模块：DBDM
    model = DBDM(in_channels=64, out_channels=128).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")
