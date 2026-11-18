"""ResNet34 + MSEC(多尺度边缘-上下文门控残差) —— 2026-10-23

设计依据(见 experiments/ 与对话分析):
  误差集中 Benign<->Malignant 混淆 => 需要局部/方向性/多尺度的组织结构线索;
  SE 失败教训 = 随机初始化的常开门控破坏预训练校准 => MSEC 用零初始化 gamma,
  epoch0 网络严格等于 vanilla ResNet34, 模块仅在有益时才生长;
  灰度复制3通道 => 通道注意力信号弱, 空间/边缘结构才是有效维度。

结构(每个插入点, ch 通道):
  r = ReLU(1x1 reduce ch->ch/4)
  ├ A 方向条带(DEGConv/MSInit 血统): dw(1xk->kx1) ∥ dw(kx1->1xk) ∥ dw3x3, concat->1x1
  ├ B 冻结Scharr边缘先验(SFFEM 血统): |gx|∥|gy| 深度卷积(不可学习) -> 1x1(可学习)
  └ C 多尺度膨胀(LaSEA 血统): dw3x3 d=2 ∥ d=4, concat -> 1x1
  fused = BN(1x1(concat(A,B,C)) -> ch)
  out = x + gamma * fused      (gamma 逐通道零初始化)

插入位置: layer3/layer4 出口(_p234 变体加 layer2)。全程深度可分离, 每点 ~0.1-0.4M 参数。
预训练加载: backbone 参数名不变, 白名单段 {"msec"}。

注册名: resnet34_msec_v1(全量) / _strips(A) / _edge(B) / _dil(C) / _v1_p234(2/3/4层)
"""
import torch
import torch.nn as nn


def _dw(ch, k, dilation=1):
    if isinstance(k, int):
        k = (k, k)
    pad = ((k[0] - 1) // 2 * dilation, (k[1] - 1) // 2 * dilation)
    return nn.Conv2d(ch, ch, kernel_size=k, stride=1, padding=pad,
                     dilation=(dilation, dilation), groups=ch, bias=False)


class _FrozenScharr(nn.Module):
    """逐通道固定 Scharr 梯度(不可学习), 返回 |gx| 与 |gy| 的拼接。"""

    _KX = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]
    _KY = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]

    def __init__(self, ch):
        super().__init__()
        wx = torch.tensor(self._KX, dtype=torch.float32).expand(ch, 1, 3, 3).clone()
        wy = torch.tensor(self._KY, dtype=torch.float32).expand(ch, 1, 3, 3).clone()
        self.gx = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.gy = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        with torch.no_grad():
            self.gx.weight.copy_(wx)
            self.gy.weight.copy_(wy)
        self.gx.weight.requires_grad = False
        self.gy.weight.requires_grad = False

    def forward(self, x):
        return torch.cat([self.gx(x).abs(), self.gy(x).abs()], dim=1)


class MSECGate(nn.Module):
    """零初始化门控残差: A 方向条带 + B 冻结Scharr + C 多尺度膨胀。"""

    def __init__(self, ch, k=7, use_strips=True, use_edge=True, use_dil=True):
        super().__init__()
        assert use_strips or use_edge or use_dil
        self.use_strips, self.use_edge, self.use_dil = use_strips, use_edge, use_dil
        r = max(ch // 4, 16)
        self.reduce = nn.Sequential(nn.Conv2d(ch, r, 1, bias=False), nn.BatchNorm2d(r), nn.ReLU(inplace=True))
        parts = []
        if use_strips:
            self.strip_h = nn.Sequential(_dw(r, (1, k)), _dw(r, (k, 1)))
            self.strip_v = nn.Sequential(_dw(r, (k, 1)), _dw(r, (1, k)))
            self.mid = _dw(r, 3)
            self.strip_fuse = nn.Conv2d(r * 3, r, 1, bias=False)
            parts.append(r)
        if use_edge:
            self.scharr = _FrozenScharr(r)
            self.edge_fuse = nn.Conv2d(r * 2, r, 1, bias=False)
            parts.append(r)
        if use_dil:
            self.dil2 = _dw(r, 3, dilation=2)
            self.dil4 = _dw(r, 3, dilation=4)
            self.dil_fuse = nn.Conv2d(r * 2, r, 1, bias=False)
            parts.append(r)
        self.fuse = nn.Conv2d(sum(parts), ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.gamma = nn.Parameter(torch.zeros(ch))

    def forward(self, x):
        r0 = self.reduce(x)
        feats = []
        if self.use_strips:
            feats.append(self.strip_fuse(torch.cat(
                [self.strip_h(r0), self.strip_v(r0), self.mid(r0)], dim=1)))
        if self.use_edge:
            feats.append(self.edge_fuse(self.scharr(r0)))
        if self.use_dil:
            feats.append(self.dil_fuse(torch.cat([self.dil2(r0), self.dil4(r0)], dim=1)))
        fused = self.bn(self.fuse(torch.cat(feats, dim=1)))
        return x + self.gamma.view(1, -1, 1, 1) * fused


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


class ResNetMSEC(nn.Module):
    """vanilla ResNet + ModuleDict('msec') 在指定 stage 出口插入 MSECGate。"""

    def __init__(self, blocks_num, num_classes=1000, include_top=True,
                 positions=(3, 4), use_strips=True, use_edge=True, use_dil=True):
        super().__init__()
        self.include_top = include_top
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
        self.positions = positions
        self.msec = nn.ModuleDict({
            str(p): MSECGate(ch_map[p], use_strips=use_strips, use_edge=use_edge, use_dil=use_dil)
            for p in positions
        })
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

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        if 2 in self.positions:
            x = self.msec["2"](x)
        x = self.layer3(x)
        if 3 in self.positions:
            x = self.msec["3"](x)
        x = self.layer4(x)
        if 4 in self.positions:
            x = self.msec["4"](x)
        if self.include_top:
            x = self.fc(torch.flatten(self.avgpool(x), 1))
        return x


def resnet34_msec_v1(num_classes=1000, include_top=True):
    # 全量 A+B+C @ layer3/4
    return ResNetMSEC([3, 4, 6, 3], num_classes, include_top)


def resnet34_msec_strips(num_classes=1000, include_top=True):
    return ResNetMSEC([3, 4, 6, 3], num_classes, include_top, use_edge=False, use_dil=False)


def resnet34_msec_edge(num_classes=1000, include_top=True):
    return ResNetMSEC([3, 4, 6, 3], num_classes, include_top, use_strips=False, use_dil=False)


def resnet34_msec_dil(num_classes=1000, include_top=True):
    return ResNetMSEC([3, 4, 6, 3], num_classes, include_top, use_strips=False, use_edge=False)


def resnet34_msec_v1_p234(num_classes=1000, include_top=True):
    return ResNetMSEC([3, 4, 6, 3], num_classes, include_top, positions=(2, 3, 4))
