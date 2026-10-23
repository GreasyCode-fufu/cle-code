import torch
import torch.nn as nn
import torch.nn.functional as F


class ECAChannelGate(nn.Module):
    """Efficient channel attention with average and max descriptors."""

    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)
        max_value = self.max_pool(x)
        descriptor = 0.5 * (avg + max_value)
        weight = self.conv(descriptor.squeeze(-1).transpose(-1, -2))
        return self.sigmoid(weight.transpose(-1, -2).unsqueeze(-1))


class TopologyGate(nn.Module):
    """Local continuity gate for boundary- and crack-like structures."""

    def __init__(self, channel, hidden=None):
        super().__init__()
        hidden = hidden or max(8, channel // 4)
        self.project = nn.Sequential(
            nn.Conv2d(channel + 4, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        base = x.mean(dim=1, keepdim=True)
        dx = (base[:, :, :, 1:] - base[:, :, :, :-1]).abs()
        dy = (base[:, :, 1:, :] - base[:, :, :-1, :]).abs()
        dxy = (base[:, :, 1:, 1:] - base[:, :, :-1, :-1]).abs()
        dyx = (base[:, :, 1:, :-1] - base[:, :, :-1, 1:]).abs()

        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        dxy = F.pad(dxy, (0, 1, 0, 1))
        dyx = F.pad(dyx, (1, 0, 0, 1))
        continuity = torch.cat([dx, dy, dxy, dyx], dim=1)
        return self.project(torch.cat([x, continuity], dim=1))


class TopoGateDBFE(nn.Module):
    """Topology-gated DBFE for texture-heavy semantic segmentation.

    It preserves the original DBFE interface while adding a lightweight
    continuity gate. The gate emphasizes locally connected defect structures
    before the Hilbert/topology-aware decoder computes category correlations.
    """

    def __init__(self, channel, k_size=3, dropout=0.1):
        super().__init__()
        self.channel_gate = ECAChannelGate(channel, k_size=k_size)
        self.topology_gate = TopologyGate(channel)
        self.local_3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False),
            nn.BatchNorm2d(channel),
            nn.GELU(),
            nn.Conv2d(channel, channel, kernel_size=1, bias=False),
        )
        self.directional = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1), groups=channel, bias=False),
            nn.Conv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0), groups=channel, bias=False),
            nn.BatchNorm2d(channel),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        channel_weight = self.channel_gate(x)
        topology_weight = self.topology_gate(x)

        local = self.local_3x3(x)
        directional = self.directional(x)
        fused = self.fuse(torch.cat([local, directional], dim=1))

        enhanced = fused * channel_weight
        topology_residual = x * topology_weight
        return enhanced + self.res_scale.tanh() * topology_residual


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(2, 64, 128, 128, device=device)
    model = TopoGateDBFE(64).to(device)
    output_tensor = model(input_tensor)

    print(model)
    print("input_tensor_shape :", input_tensor.shape)
    print("output_tensor_shape:", output_tensor.shape)
    print("哔哩哔哩/微信公众号 CV缝合救星 独家整理!")
