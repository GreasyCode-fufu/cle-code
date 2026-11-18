"""Baseline 主干再设计变体 —— 2026-10-24(用户: baseline 也可再设计)

- resnet34_gmp: 分类头 GAP -> GMP(全局最大池化)。纹理判别任务中 GMP 保留最强
  响应区域, 与 GAP 互补; 零新增参数, ImageNet 直接载入(键不变)。
- resnet34_dshortcut: ResNet-D 式短接 —— stride2 的 1x1 卷积短接改为
  AvgPool(2)+1x1(s1)。抗混叠、保留信息(经典涨点结构, torchvision 新版默认)。
  参数形状不变, 仅键名重排(downsample.0/.1 -> downsample.conv/.bn),
  由 MODEL_CHECKPOINT_TRANSFORMS 做键名重映射后严格载入, 白名单 {"downsample_avg"}。
"""
import torch
import torch.nn as nn


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


def _dshortcut(in_c, out_c, stride):
    """AvgPool + 1x1(s1) + BN, 命名 conv/bn(由变换重映射后与 checkpoint 对齐)。"""
    from collections import OrderedDict
    return nn.Sequential(OrderedDict([
        ("downsample_avg", nn.AvgPool2d(2, stride=2, ceil_mode=True)),
        ("conv", nn.Conv2d(in_c, out_c, 1, stride=1, bias=False)),
        ("bn", nn.BatchNorm2d(out_c)),
    ]))


class ResNetRedesign(nn.Module):
    def __init__(self, blocks_num, num_classes=1000, include_top=True,
                 dshortcut=False, gmp_head=False):
        super().__init__()
        self.include_top = include_top
        self.gmp_head = gmp_head
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._mk(64, blocks_num[0], 1, dshortcut)
        self.layer2 = self._mk(128, blocks_num[1], 2, dshortcut)
        self.layer3 = self._mk(256, blocks_num[2], 2, dshortcut)
        self.layer4 = self._mk(512, blocks_num[3], 2, dshortcut)
        if include_top:
            self.avgpool = nn.AdaptiveMaxPool2d((1, 1)) if gmp_head else nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _mk(self, ch, n, stride, dshortcut):
        ds = None
        if stride != 1 or self.in_channel != ch:
            ds = _dshortcut(self.in_channel, ch, stride) if dshortcut else nn.Sequential(
                nn.Conv2d(self.in_channel, ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(ch))
        layers = [BasicBlock(self.in_channel, ch, stride=stride, downsample=ds)]
        self.in_channel = ch
        layers += [BasicBlock(ch, ch) for _ in range(n - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.fc(torch.flatten(self.avgpool(x), 1))
        return x


def resnet34_gmp(num_classes=1000, include_top=True):
    return ResNetRedesign([3, 4, 6, 3], num_classes, include_top, gmp_head=True)


def resnet34_dshortcut(num_classes=1000, include_top=True):
    return ResNetRedesign([3, 4, 6, 3], num_classes, include_top, dshortcut=True)
