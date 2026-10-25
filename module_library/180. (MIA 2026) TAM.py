import torch
import torch.nn as nn


class TAM(nn.Module):
    """
    Temporal Attention Module for motion-guided cardiac segmentation.

    Input : frames with shape [B, C, H, W, T]
    Output: refined temporal features with shape [B, C, H, W, T]
    """

    def __init__(self, all_channels=256, embedding_dim=256, num_heads=8, dropout=0.0):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.all_channels = all_channels
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

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

        # 时空自门控：给每个时间帧的 attention summary 生成空间置信图
        self.gate_conv = nn.Conv2d(all_channels, 1, kernel_size=1, bias=True)
        self.gate_activation = nn.Sigmoid()

        # attention 结果和原始帧特征融合，保持即插即用接口不变
        self.combine_conv = nn.Conv2d(all_channels * 2, all_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.LayerNorm(all_channels)
        self.activation = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(all_channels, all_channels, kernel_size=1, bias=True)

    def forward(self, frames):
        if frames.ndim != 5:
            raise ValueError("TAM expects a 5D tensor: [B, C, H, W, T].")

        B, C, H, W, T = frames.shape
        if C != self.all_channels:
            raise ValueError(f"Input channels {C} do not match all_channels={self.all_channels}.")

        # 每个空间位置都形成一条长度为 T 的时间序列，然后做多头跨时间注意力
        x = frames.permute(0, 4, 2, 3, 1).reshape(B, T, H * W, C)
        x = self.in_proj(x)
        x = x.permute(0, 2, 1, 3).reshape(B * H * W, T, self.embedding_dim)

        attn_output, _ = self.cross_time_attention(x, x, x, need_weights=False)
        attn_output = attn_output.reshape(B, H * W, T, self.embedding_dim).permute(0, 2, 1, 3)
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.reshape(B, T, H, W, C).permute(0, 4, 2, 3, 1)

        refined_frames = []
        for t in range(T):
            attn_t = attn_output[..., t]
            gate_t = self.gate_activation(self.gate_conv(attn_t))
            gated_t = attn_t * gate_t

            fused = torch.cat([frames[..., t], gated_t], dim=1)
            fused = self.combine_conv(fused)
            fused = fused.permute(0, 2, 3, 1)
            fused = self.norm(fused)
            fused = fused.permute(0, 3, 1, 2)
            fused = self.activation(fused)
            refined_frames.append(self.classifier(fused + frames[..., t]))

        return torch.stack(refined_frames, dim=-1)


TemporalAttention = TAM


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟 2D+t 心脏分割特征输入 [B, C, H, W, T]
    input_tensor = torch.randn(1, 256, 16, 16, 10).to(device)

    # 直接实例化真正的核心创新模块：TAM
    model = TAM(all_channels=256, embedding_dim=256, num_heads=8, dropout=0.1).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")
