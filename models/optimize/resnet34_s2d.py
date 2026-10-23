"""ResNet34 + space_to_depth 变体(v1 / v2 双版本) —— 2026-10-23

同一方法的两个设计版本, 注册名带版本号, 训练权重目录自动分离:
  ./weights/resnet34_s2d_v1/   ./weights/resnet34_s2d_v2/

v1 (后置版, 已有训练权重):
    conv1(7x7,s2, 3->64) -> bn1 -> relu
    -> space_to_depth(64->256ch@56x56) -> stem_adapter(1x1 256->64 + BN + ReLU) -> layer1
    问题: conv1 的 s2 已经丢弃一半像素, s2d 无从挽回; 随机初始化的压缩卷积是瓶颈。
    预训练加载: 白名单放行 stem_adapter (train_cle_fair.MODEL_ALLOWED_MISSING_SEGMENTS)。

v2 (前置版, Focus/SPD-Conv 式, 当前推荐):
    space_to_depth(3->12ch@112x112) -> conv1(7x7, s1, 12->64) -> bn1 -> relu
    -> maxpool -> layer1
    全部像素参与第一层特征提取, 4 相位成为通道由卷积学习融合。
    预训练加载: MODEL_CHECKPOINT_TRANSFORMS 将 conv1 预训练核复制4份/除4膨胀为12通道。

两版 layer1~layer4/fc 完全一致, 空间尺寸流一致(layer1 输入 64ch@56x56)。

注册名: resnet34_s2d_v1 / resnet34_s2d_v2 (cle_fair_common.MODEL_REGISTRY 与 evaluate.py)
"""
import torch
import torch.nn as nn


class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                          x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class _StemBase(nn.Module):
    """layer1~layer4/fc 与 models/Resnet.py 完全一致, 子类只定义 stem。"""

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super().__init__()
        self.include_top = include_top
        self.in_channel = 64

        self._build_stem()
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _build_stem(self):
        raise NotImplementedError

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def _forward_stem(self, x):
        raise NotImplementedError

    def forward(self, x):
        x = self._forward_stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


class ResNetS2Dv1(_StemBase):
    """v1 后置版: s2d + 压缩卷积替代 maxpool(位于 conv1 之后)。"""

    def _build_stem(self):
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.space_to_depth = space_to_depth()
        self.stem_adapter = nn.Sequential(
            nn.Conv2d(self.in_channel * 4, self.in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def _forward_stem(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.space_to_depth(x)  # (B,64,H,W) -> (B,256,H/2,W/2) 无损
        x = self.stem_adapter(x)    # (B,256,H/2,W/2) -> (B,64,H/2,W/2)
        return x


class ResNetS2Dv2(_StemBase):
    """v2 前置版(Focus/SPD-Conv 式): s2d 在 conv1 之前, conv1 步距 2->1。"""

    def _build_stem(self):
        self.space_to_depth = space_to_depth()  # (B,3,H,W) -> (B,12,H/2,W/2)
        self.conv1 = nn.Conv2d(12, self.in_channel, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _forward_stem(self, x):
        x = self.space_to_depth(x)  # 全像素保留
        x = self.conv1(x)           # s1: 12->64, 输出 H/2 x W/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


def resnet34_s2d_v1(num_classes=1000, include_top=True):
    # ImageNet 权重: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNetS2Dv1(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet34_s2d_v2(num_classes=1000, include_top=True):
    # ImageNet 权重: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # (conv1 由加载器膨胀: 3ch 核复制4份/除4 -> 12ch)
    return ResNetS2Dv2(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
