import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        act=True,
        norm=True,
    ):
        super().__init__()
        if padding is None:
            padding = tuple(k // 2 for k in kernel_size) if isinstance(kernel_size, tuple) else kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=not norm,
        )
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HaarDWT2D(nn.Module):
    """Depthwise 2D Haar wavelet decomposition."""

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) * 0.5
        lh = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) * 0.5
        hl = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) * 0.5
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) * 0.5
        self.register_buffer("filters", torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1))

    def forward(self, x):
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        weight = self.filters.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, padding=0, groups=c)
        y = y.view(b, c, 4, y.shape[-2], y.shape[-1])
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3], (h, w)


class HaarIDWT2D(nn.Module):
    """Inverse depthwise 2D Haar wavelet reconstruction."""

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
        y = F.conv_transpose2d(x, weight, stride=2, padding=0, groups=c)
        if out_size is not None:
            out_h, out_w = out_size
            y = y[:, :, :out_h, :out_w]
        return y


class REM(nn.Module):
    """Radar Enhancement Module with wavelet subband decoupling."""

    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        use_residual=True,
        norm=True,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or out_channels
        self.use_residual = use_residual

        self.in_proj = ConvBNAct(in_channels, hidden_channels, kernel_size=1, padding=0, norm=norm)
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        self.ll_branch = ConvBNAct(hidden_channels, hidden_channels, kernel_size=1, padding=0, norm=norm)
        self.lh_branch = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=(1, 3), norm=norm),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=(3, 1), norm=norm),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, norm=norm),
        )
        self.hl_branch = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=(3, 1), norm=norm),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=(1, 3), norm=norm),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, norm=norm),
        )
        self.hh_branch = ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, norm=norm)
        self.out_proj = ConvBNAct(hidden_channels, out_channels, kernel_size=1, padding=0, act=False, norm=norm)

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else ConvBNAct(in_channels, out_channels, kernel_size=1, padding=0, act=False, norm=norm)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.in_proj(x)
        ll, lh, hl, hh, out_size = self.dwt(x)

        ll = self.ll_branch(ll)
        lh = self.lh_branch(lh)
        hl = self.hl_branch(hl)
        hh = self.hh_branch(hh)

        x = self.idwt(ll, lh, hl, hh, out_size=out_size)
        x = self.out_proj(x)
        return x + identity if self.use_residual else x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(2, 64, 128, 128, device=device)
    model = REM(64, 64).to(device)
    output_tensor = model(input_tensor)

    print(model)
    print("input_tensor_shape :", input_tensor.shape)
    print("output_tensor_shape:", output_tensor.shape)
    print("哔哩哔哩/微信公众号 CV缝合救星 独家整理!")
