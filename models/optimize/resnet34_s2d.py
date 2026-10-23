"""ResNet34 + space_to_depth 下采样变体(方案B) —— 2026-10-23 新增

改造点(仅 stem 的下采样方式, 其余与 models/Resnet.py 完全一致):

  原版:  conv1(7x7,s2) -> bn1 -> relu -> maxpool(3x3,s2) -> layer1
  本版:  conv1(7x7,s2) -> bn1 -> relu -> space_to_depth -> stem_adapter(1x1压缩) -> layer1

  - space_to_depth: 空间到深度无损重排(64ch@112x112 -> 256ch@56x56),
    不丢弃任何像素(原版 maxpool 会丢弃 3/4 空间信息);
  - stem_adapter: 1x1 卷积 256->64 + BN + ReLU, 把重排后的通道压回 layer1
    期望的 64 通道(新增模块, ImageNet 预训练加载时按白名单放行, 随机初始化);
  - 空间尺寸流与原版严格一致: layer1 依旧收到 64ch@56x56;
  - conv1/bn1/layer1~layer4/fc 的参数名与形状完全不变 -> ImageNet 权重
    (resnet34-333f7ec4.pth) 除 fc 与 stem_adapter 外全部严格载入。

注册名: resnet34_s2d (cle_fair_common.MODEL_REGISTRY 与 evaluate.py)
白名单: train_cle_fair.py MODEL_ALLOWED_MISSING_SEGMENTS -> {"stem_adapter"}
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


class ResNet(nn.Module):
    """与 models/Resnet.py 的 ResNet 唯一区别: stem 用 s2d+adapter 替代 maxpool。"""

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # ↓ 替代 maxpool: 无损空间重排 + 可学习通道压缩
        self.space_to_depth = space_to_depth()
        self.stem_adapter = nn.Sequential(
            nn.Conv2d(self.in_channel * 4, self.in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.space_to_depth(x)  # (B,64,H,W) -> (B,256,H/2,W/2) 无损
        x = self.stem_adapter(x)    # (B,256,H/2,W/2) -> (B,64,H/2,W/2)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34_s2d(num_classes=1000, include_top=True):
    # ImageNet 权重: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
