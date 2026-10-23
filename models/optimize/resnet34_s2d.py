"""ResNet34 + space_to_depth 前置变体(方案B-v2, Focus/SPD-Conv 式) —— 2026-10-23 重设计

v1(已废弃)把 s2d 放在 conv1 之后, 但 conv1(7x7,s2)本身已丢弃一半像素,
s2d 无从挽回; 且随机初始化的 256->64 压缩卷积形成未训练瓶颈。
v2 将 s2d 前置到 conv1 之前 —— 网络第一层特征提取即看到全部像素:

  原版:  conv1(7x7,s2, 3->64)  -> bn1 -> relu -> maxpool -> layer1
  v2 :   space_to_depth(3->12ch@112x112)
         -> conv1(7x7, s1, 12->64) -> bn1 -> relu -> maxpool -> layer1

  - 224x224 的每个像素都参与 conv1 特征提取(原版 s2 卷积隔行采样);
  - 4 个相位子图成为 4 组输入通道, 由卷积学习融合(YOLOv5 Focus / SPD-Conv 标准做法);
  - conv1 步距 2->1, 输出尺寸 112x112 与原版一致; maxpool 后 layer1 仍为 64ch@56x56;
  - layer1~layer4/fc 参数名与形状不变;
  - conv1 输入通道 3->12: ImageNet 预训练核在 train_cle_fair.py 的
    MODEL_CHECKPOINT_TRANSFORMS 中膨胀(复制4份/除4)后载入;
  - 无新增可学习模块, 白名单不再需要(missing 仅 fc)。

注册名: resnet34_s2d (cle_fair_common.MODEL_REGISTRY 与 evaluate.py)
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
    """与 models/Resnet.py 的 ResNet 唯一区别: stem 前置 space_to_depth, conv1 步距 2->1。"""

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        # ↓ 无损下采样前置: (B,3,224,224) -> (B,12,112,112)
        self.space_to_depth = space_to_depth()
        self.conv1 = nn.Conv2d(12, self.in_channel, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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
        x = self.space_to_depth(x)  # (B,3,H,W) -> (B,12,H/2,W/2) 全像素保留
        x = self.conv1(x)           # s1: 12->64, 输出 H/2 x W/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

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
    # (conv1 由加载器膨胀: 3ch 核复制4份/除4 -> 12ch)
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
