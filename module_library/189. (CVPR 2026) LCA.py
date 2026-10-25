import torch
import torch.nn as nn


class LightweightChannelAttention(nn.Module):
    """Pooling-free per-pixel, per-channel gate from RMAE-ProGRess."""

    def __init__(self, channels: int):
        super().__init__()
        self.channel_mixer = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.sigmoid(self.channel_mixer(x))
        return x * gate


class LCAR(nn.Module):
    """Lightweight Channel Attention with an optional residual path."""

    def __init__(self, channels: int, residual: bool = True):
        super().__init__()
        self.lca = LightweightChannelAttention(channels)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lca(x)
        if self.residual:
            y = y + x
        return y


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 128, 128, device=device)
    model = LCAR(64, residual=True).to(device)
    output_tensor = model(input_tensor)

    print(model)
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号 CV缝合救星 独家整理!\n")
