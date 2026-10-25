import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CardioMotionTAM(nn.Module):
    """
    CardioMotion-TAM: cardiac-cycle aware temporal attention module.

    It keeps the original TAM interface:
    input/output shape = [B, C, H, W, T].
    """

    def __init__(self, all_channels=256, embedding_dim=256, num_heads=8, dropout=0.0, max_frames=32):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.all_channels = all_channels
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.max_frames = max_frames

        self.in_proj = (
            nn.Linear(all_channels, embedding_dim, bias=False)
            if embedding_dim != all_channels
            else nn.Identity()
        )
        self.out_proj = (
            nn.Linear(embedding_dim, all_channels, bias=False)
            if embedding_dim != all_channels
            else nn.Identity()
        )
        self.cross_time_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        # 心动周期相位偏置：用可学习时序位置表征辅助跨帧匹配
        self.phase_embed = nn.Parameter(torch.zeros(1, max_frames, embedding_dim))
        nn.init.trunc_normal_(self.phase_embed, std=0.02)

        # 运动差异门控：显式使用相邻帧差异，抑制噪声帧、强化真实运动边界
        self.motion_gate = nn.Sequential(
            nn.Conv2d(all_channels + 1, all_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(all_channels // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(all_channels // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.temporal_confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(all_channels, max(all_channels // 8, 8), kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(all_channels // 8, 8), all_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.combine_conv = nn.Conv2d(all_channels * 2, all_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.LayerNorm(all_channels)
        self.activation = nn.SiLU(inplace=True)
        self.classifier = nn.Conv2d(all_channels, all_channels, kernel_size=1, bias=True)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def _phase_bias(self, T, device):
        if T <= self.max_frames:
            return self.phase_embed[:, :T].to(device)

        # 当时间帧数量超过预设长度时，用线性插值扩展相位表征
        table = self.phase_embed.transpose(1, 2)
        table = F.interpolate(table, size=T, mode="linear", align_corners=False)
        return table.transpose(1, 2).to(device)

    def _motion_energy(self, frames, t):
        B, C, H, W, T = frames.shape
        prev_t = max(t - 1, 0)
        next_t = min(t + 1, T - 1)
        diff = (frames[..., t] - 0.5 * (frames[..., prev_t] + frames[..., next_t])).abs()
        energy = diff.mean(dim=1, keepdim=True)
        energy = energy / (energy.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return diff, energy

    def forward(self, frames):
        if frames.ndim != 5:
            raise ValueError("CardioMotionTAM expects a 5D tensor: [B, C, H, W, T].")

        B, C, H, W, T = frames.shape
        if C != self.all_channels:
            raise ValueError(f"Input channels {C} do not match all_channels={self.all_channels}.")

        x = frames.permute(0, 4, 2, 3, 1).reshape(B, T, H * W, C)
        x = self.in_proj(x)
        x = x + self._phase_bias(T, x.device).unsqueeze(2)
        x = x.permute(0, 2, 1, 3).reshape(B * H * W, T, self.embedding_dim)

        attn_output, _ = self.cross_time_attention(x, x, x, need_weights=False)
        attn_output = attn_output.reshape(B, H * W, T, self.embedding_dim).permute(0, 2, 1, 3)
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.reshape(B, T, H, W, C).permute(0, 4, 2, 3, 1)

        refined_frames = []
        for t in range(T):
            attn_t = attn_output[..., t]
            diff_t, motion_energy = self._motion_energy(frames, t)
            spatial_gate = self.motion_gate(torch.cat([attn_t, motion_energy], dim=1))
            channel_gate = self.temporal_confidence(diff_t)
            gated_t = attn_t * spatial_gate * channel_gate

            fused = torch.cat([frames[..., t], gated_t], dim=1)
            fused = self.combine_conv(fused)
            fused = fused.permute(0, 2, 3, 1)
            fused = self.norm(fused)
            fused = fused.permute(0, 3, 1, 2)
            fused = self.activation(fused)
            fused = self.classifier(fused + frames[..., t])
            refined_frames.append(fused + self.res_scale * diff_t)

        return torch.stack(refined_frames, dim=-1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟 2D+t 心脏分割特征输入 [B, C, H, W, T]
    input_tensor = torch.randn(1, 256, 16, 16, 10).to(device)

    # 直接实例化真正的核心创新模块：CardioMotionTAM
    model = CardioMotionTAM(all_channels=256, embedding_dim=256, num_heads=8, dropout=0.1).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")
