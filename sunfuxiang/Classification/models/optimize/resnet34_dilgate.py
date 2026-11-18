"""Batch C 变体: DilGate(纯多膨胀零初始化门控) + PFESA(免参数FFT边缘/结构注意力)

依据 Batch B 消融: msec_dil(test 0.8986) 唯一显著超过 vanilla(0.8843),
条带/边缘分支为稀释项 => 本文件只保留膨胀分支并深挖;
另引入 PFESA(MICCAI 2026, 全免参数) 及其与 DilGate 的组合。

零初始化 gamma 语义与 MSEC 相同: epoch0 严格等于 vanilla。
PFESA 无任何可学习参数, 仅对特征做频域高低频注意力重标定(为 AMP 兼容,
fft 计算前转 float32, 结果转回输入 dtype)。

注册名:
  resnet34_msec_dil_p234  DilGate(d=2,4) @ layer2/3/4
  resnet34_msec_dil234    DilGate(d=2,3,4) @ layer3/4
  resnet34_pfesa          PFESA @ layer3/4 (零新增参数)
  resnet34_pfesa_dil      PFESA + DilGate(d=2,4) @ layer3/4
白名单: {"gates"}(DilGate 系列); pfesa 系列无新增参数, 无需白名单。
"""
import torch
import torch.nn as nn
import torch.fft as fft


def _dw(ch, k, dilation=1):
    return nn.Conv2d(ch, ch, kernel_size=k, stride=1,
                     padding=(k - 1) // 2 * dilation, dilation=dilation,
                     groups=ch, bias=False)


class DilGate(nn.Module):
    """纯多尺度膨胀 + 零初始化门控残差。"""

    def __init__(self, ch, rates=(2, 4), r_div=4, k=3, scalar_gamma=False, use_identity=False):
        super().__init__()
        r = max(ch // r_div, 16)
        self.scalar_gamma = scalar_gamma
        self.use_identity = use_identity
        self.reduce = nn.Sequential(
            nn.Conv2d(ch, r, 1, bias=False), nn.BatchNorm2d(r), nn.ReLU(inplace=True))
        self.dils = nn.ModuleList([_dw(r, k, d) for d in rates])
        if use_identity:
            self.dil_fuse = nn.Conv2d(r * (len(rates) + 1), ch, 1, bias=False)
        else:
            self.dil_fuse = nn.Conv2d(r * len(rates), ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.gamma = nn.Parameter(torch.zeros(1 if scalar_gamma else ch))

    def forward(self, x):
        r0 = self.reduce(x)
        f = torch.cat([d(r0) for d in self.dils], dim=1)
        if self.use_identity:
            f = torch.cat([f, r0], dim=1)
        g = self.gamma.view(1, -1, 1, 1) if not self.scalar_gamma else self.gamma.view(1, 1, 1, 1)
        return x + g * self.bn(self.dil_fuse(f))


class PFESA(nn.Module):
    """免参数边缘-结构注意力(MICCAI 2026)。高低频分离->边缘/结构注意力->特征重标定。"""

    def __init__(self, base_ratio=0.1):
        super().__init__()
        self.base_ratio = base_ratio
        self.eps = 1e-5

    def _mask(self, h, w, device):
        ratio = self.base_ratio * min(h, w) / max(h, w)
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        return torch.exp(-(gy ** 2 + gx ** 2) / (2 * ratio ** 2))

    def forward(self, feat):
        inp = feat
        x = feat.float()  # AMP 下 fft 需要 float32
        B, C, H, W = x.shape
        spec = fft.fftshift(fft.fftn(x, dim=(-2, -1)), dim=(-2, -1))
        m = self._mask(H, W, x.device)
        low = torch.abs(fft.ifftn(spec * m, dim=(-2, -1)))
        high = torch.abs(fft.ifftn(spec * (1 - m), dim=(-2, -1)))
        # 结构注意力(低频能量归一化)
        e = low.pow(2)
        s_attn = (e - e.mean(dim=[2, 3], keepdim=True)) / (e.var(dim=[2, 3], keepdim=True) + self.eps)
        # 边缘注意力(高频方差归一化的平方偏差)
        dev = (high - high.mean(dim=[2, 3], keepdim=True)).pow(2)
        e_attn = dev / (high.var(dim=[2, 3], keepdim=True) + self.eps)
        attn = torch.sigmoid(s_attn + e_attn)
        return inp * attn.to(inp.dtype)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetDilPFESA(nn.Module):
    """vanilla ResNet + 可选 PFESA / 可选 DilGate 于指定 stage 出口。"""

    def __init__(self, blocks_num, num_classes=1000, include_top=True,
                 positions=(3, 4), rates=(2, 4), use_pfesa=False, use_dil=True,
                 r_div=4, k=3, scalar_gamma=False, rates_map=None, double_gate=False, use_identity=False):
        super().__init__()
        self.include_top = include_top
        self.positions = positions
        self.use_pfesa, self.use_dil = use_pfesa, use_dil
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks_num[0])
        self.layer2 = self._make_layer(128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(512, blocks_num[3], stride=2)
        ch_map = {2: 128, 3: 256, 4: 512}
        self.pfesa = nn.ModuleDict({str(p): PFESA() for p in positions}) if use_pfesa else None
        def _mk(p):
            kw = dict(rates=(rates_map or {}).get(p, rates), r_div=r_div, k=k,
                      scalar_gamma=scalar_gamma, use_identity=use_identity)
            g = DilGate(ch_map[p], **kw)
            return nn.Sequential(g, DilGate(ch_map[p], **kw)) if double_gate else g
        self.gates = nn.ModuleDict({str(p): _mk(p) for p in positions}) if use_dil else None
        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channel))
        layers = [BasicBlock(self.in_channel, channel, downsample=downsample, stride=stride)]
        self.in_channel = channel
        layers += [BasicBlock(channel, channel) for _ in range(1, block_num)]
        return nn.Sequential(*layers)

    def _augment(self, x, p):
        if self.pfesa is not None:
            x = self.pfesa[str(p)](x)
        if self.gates is not None:
            x = self.gates[str(p)](x)
        return x

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        if 2 in self.positions:
            x = self._augment(x, 2)
        x = self.layer3(x)
        if 3 in self.positions:
            x = self._augment(x, 3)
        x = self.layer4(x)
        if 4 in self.positions:
            x = self._augment(x, 4)
        if self.include_top:
            x = self.fc(torch.flatten(self.avgpool(x), 1))
        return x


def resnet34_dil234w(num_classes=1000, include_top=True):
    # dil234 加宽版: reduce 通道 ch/2
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(3, 4), rates=(2, 3, 4), r_div=2)


def resnet34_dil2346(num_classes=1000, include_top=True):
    # 四膨胀率 2/3/4/6
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(3, 4), rates=(2, 3, 4, 6))


def resnet34_dil123(num_classes=1000, include_top=True):
    # 膨胀率 1/2/3(含原生尺度)
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(3, 4), rates=(1, 2, 3))


def resnet34_dil_r8(num_classes=1000, include_top=True):
    # 更窄 reduce (ch/8)
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(3, 4), rates=(2, 3, 4), r_div=8)


def resnet34_dil234_l4(num_classes=1000, include_top=True):
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(4,), rates=(2, 3, 4))


def resnet34_dil234_l3(num_classes=1000, include_top=True):
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(3,), rates=(2, 3, 4))


def resnet34_dil234_k5(num_classes=1000, include_top=True):
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(3, 4), rates=(2, 3, 4), k=5)


def resnet34_dil234_sg(num_classes=1000, include_top=True):
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(3, 4), rates=(2, 3, 4), scalar_gamma=True)


def resnet34_dil23_r(num_classes=1000, include_top=True):
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(3, 4), rates=(2, 3))


def resnet34_dil34_r(num_classes=1000, include_top=True):
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(3, 4), rates=(3, 4))


def resnet34_dil234_prog(num_classes=1000, include_top=True):
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(3, 4),
                          rates=(2, 3, 4), rates_map={3: (2, 3), 4: (3, 4)})


def resnet34_dil234_mix(num_classes=1000, include_top=True):
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(3, 4),
                          rates=(2, 3, 4), rates_map={3: (2, 3, 4), 4: (2, 4)})


def resnet34_dil234_id(num_classes=1000, include_top=True):
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(3, 4),
                          rates=(2, 3, 4), use_identity=True)


def resnet34_dil234_x2(num_classes=1000, include_top=True):
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(3, 4),
                          rates=(2, 3, 4), double_gate=True)


def resnet34_msec_dil_p234(num_classes=1000, include_top=True):
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(2, 3, 4))


def resnet34_msec_dil234(num_classes=1000, include_top=True):
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top, positions=(3, 4), rates=(2, 3, 4))


def resnet34_pfesa(num_classes=1000, include_top=True):
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top,
                          use_pfesa=True, use_dil=False)


def resnet34_pfesa_dil(num_classes=1000, include_top=True):
    return ResNetDilPFESA([3, 4, 6, 3], num_classes, include_top,
                          use_pfesa=True, use_dil=True)
