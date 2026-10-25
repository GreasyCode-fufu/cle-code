import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=None, groups=1, act=True):
        super().__init__()
        if padding is None:
            padding = tuple(k // 2 for k in kernel_size) if isinstance(kernel_size, tuple) else kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HaarDWT2D(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) * 0.5
        lh = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) * 0.5
        hl = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) * 0.5
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) * 0.5
        self.register_buffer("filters", torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1))

    def forward(self, x):
        b, c, h, w = x.shape
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect")
        weight = self.filters.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, groups=c)
        y = y.view(b, c, 4, y.shape[-2], y.shape[-1])
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3], (h, w)


class HaarIDWT2D(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) * 0.5
        lh = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) * 0.5
        hl = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) * 0.5
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) * 0.5
        self.register_buffer("filters", torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1))

    def forward(self, ll, lh, hl, hh, out_size=None):
        b, c, h, w = ll.shape
        x = torch.stack([ll, lh, hl, hh], dim=2).view(b, c * 4, h, w)
        weight = self.filters.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        y = F.conv_transpose2d(x, weight, stride=2, groups=c)
        if out_size is not None:
            y = y[:, :, : out_size[0], : out_size[1]]
        return y


class AxialStateGate(nn.Module):
    """Mamba-style axial context without external selective-scan dependencies."""

    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def _running_mean(x, dim):
        length = x.shape[dim]
        denom = torch.arange(1, length + 1, device=x.device, dtype=x.dtype)
        shape = [1] * x.ndim
        shape[dim] = length
        return x.cumsum(dim=dim) / denom.view(*shape)

    def forward(self, x):
        h_forward = self._running_mean(x, dim=2)
        h_backward = torch.flip(self._running_mean(torch.flip(x, dims=[2]), dim=2), dims=[2])
        w_forward = self._running_mean(x, dim=3)
        w_backward = torch.flip(self._running_mean(torch.flip(x, dims=[3]), dim=3), dims=[3])
        context = torch.cat([h_forward, h_backward, w_forward, w_backward], dim=1)
        return x + self.gate(context) * self.mix(context)


class RadarPeakGate(nn.Module):
    """Sparse radar peak gate from low- and high-frequency evidence."""

    def __init__(self, channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, low_band, high_band):
        peak = high_band.abs().mean(dim=1, keepdim=True)
        floor = low_band.abs().mean(dim=1, keepdim=True)
        descriptor = torch.cat([low_band, high_band, peak, floor], dim=1)
        return self.project(descriptor)


class WaveMambaREM(nn.Module):
    """Wavelet REM with axial state context and sparse radar peak gating."""

    def __init__(self, in_channels, out_channels=None, hidden_channels=None, use_residual=True):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or out_channels
        self.use_residual = use_residual

        self.in_proj = ConvBNAct(in_channels, hidden_channels, kernel_size=1, padding=0)
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        self.ll_context = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=1, padding=0),
            AxialStateGate(hidden_channels),
        )
        self.lh_context = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=(1, 5)),
            AxialStateGate(hidden_channels),
        )
        self.hl_context = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=(5, 1)),
            AxialStateGate(hidden_channels),
        )
        self.hh_context = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, groups=hidden_channels),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=1, padding=0),
        )
        self.peak_gate = RadarPeakGate(hidden_channels)
        self.out_proj = ConvBNAct(hidden_channels, out_channels, kernel_size=1, padding=0, act=False)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else ConvBNAct(in_channels, out_channels, kernel_size=1, padding=0, act=False)
        )
        self.res_scale = nn.Parameter(torch.tensor(0.35))

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.in_proj(x)
        ll, lh, hl, hh, out_size = self.dwt(x)

        ll = self.ll_context(ll)
        lh = self.lh_context(lh)
        hl = self.hl_context(hl)
        hh = self.hh_context(hh)

        high = (lh.abs() + hl.abs() + hh.abs()) / 3.0
        gate = self.peak_gate(ll, high)
        lh = lh * gate
        hl = hl * gate
        hh = hh * gate

        x = self.idwt(ll, lh, hl, hh, out_size=out_size)
        x = self.out_proj(x)
        if self.use_residual:
            x = identity + self.res_scale.tanh() * x
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(2, 64, 128, 128, device=device)
    model = WaveMambaREM(64, 64).to(device)
    output_tensor = model(input_tensor)

    print(model)
    print("input_tensor_shape :", input_tensor.shape)
    print("output_tensor_shape:", output_tensor.shape)
    print("哔哩哔哩/微信公众号 CV缝合救星 独家整理!")
