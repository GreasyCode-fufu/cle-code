"""ResNet34 空洞化结构变体 —— 2026-10-23 结构级设计(用户开放网络结构修改权)

依据: CLE 图像判别依赖细胞/腺体级纹理细节, 标准 ResNet 在 layer3/4 连续 stride2
把特征压到 14/7 分辨率, 细节丢失; 实验也显示多尺度膨胀上下文(保持分辨率语义)有效。
空洞化(DeepLab 式): 去掉 stage 下采样, 用膨胀卷积补感受野, 深层保持更高分辨率。

- resnet34_atrous4:  layer4 stride 2->1, dilation=2 (特征 7x7 -> 14x14)
- resnet34_atrous34: layer3 s->1 d=2 + layer4 s->1 d=4 (特征 14x14 保持到 layer4)
参数名/形状与 vanilla 完全一致(stride/dilation 是前向属性), ImageNet 权重直接载入,
missing 仅 fc —— 无需白名单。
"""
import torch
import torch.nn as nn

from models.optimize.resnet34_dilgate import DilGate


class BasicBlockDil(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetAtrous(nn.Module):
    def __init__(self, blocks_num, num_classes=1000, include_top=True,
                 s3=2, d3=1, s4=2, d4=1, use_gates=False):
        super().__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks_num[0])
        self.layer2 = self._make_layer(128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(256, blocks_num[2], stride=s3, dilation=d3)
        self.layer4 = self._make_layer(512, blocks_num[3], stride=s4, dilation=d4)
        self.use_gates = use_gates
        self.gates = nn.ModuleDict({"3": DilGate(256, rates=(2, 3, 4)),
                                    "4": DilGate(512, rates=(2, 3, 4))}) if use_gates else None
        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, channel, block_num, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_channel != channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channel))
        layers = [BasicBlockDil(self.in_channel, channel, downsample=downsample,
                                stride=stride, dilation=dilation)]
        self.in_channel = channel
        layers += [BasicBlockDil(channel, channel, dilation=dilation)
                   for _ in range(1, block_num)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.use_gates:
            x = self.gates["3"](x)
        x = self.layer4(x)
        if self.use_gates:
            x = self.gates["4"](x)
        if self.include_top:
            x = self.fc(torch.flatten(self.avgpool(x), 1))
        return x


def resnet34_atrous4_dil234(num_classes=1000, include_top=True):
    return ResNetAtrous([3, 4, 6, 3], num_classes, include_top, s4=1, d4=2, use_gates=True)


def resnet34_atrous4(num_classes=1000, include_top=True):
    # layer4 不下采样, dilation=2: 最终特征 14x14(原 7x7)
    return ResNetAtrous([3, 4, 6, 3], num_classes, include_top, s4=1, d4=2)


def resnet34_atrous34(num_classes=1000, include_top=True):
    # layer3/4 都不下采样: 最终特征 28x28, 感受野由 d=2/4 补偿
    return ResNetAtrous([3, 4, 6, 3], num_classes, include_top, s3=1, d3=2, s4=1, d4=4)
