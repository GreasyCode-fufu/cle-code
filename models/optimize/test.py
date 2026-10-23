import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
#  CPCA —— Channel Prior Convolutional Attention（原始代码，未做任何改动）
#  通道注意力(avg+max) + 多尺度条带深度卷积空间注意力；self.conv 复用三次、
#  条带核固定 7/11/21，均与所给原版一致。任意特征尺寸可跑。
# =============================================================================

class CPCAChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(CPCAChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class CPCA(nn.Module):
    def __init__(self, channels, channelAttention_reduce=4):
        super().__init__()

        self.ca = CPCAChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        inputs = self.ca(inputs)

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out


class CPCAResidual(nn.Module):
    """即插即用封装：零初始化门控残差。gamma=0 时初始 output==input（保护预训练）。
    若想要"裸 CPCA"（不门控），把 forward 改成 return self.cpca(x) 即可。"""
    def __init__(self, channels: int, channelAttention_reduce: int = 4) -> None:
        super().__init__()
        self.cpca = CPCA(channels, channelAttention_reduce=channelAttention_reduce)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.cpca(x)


# =============================================================================
#  ResNet（原始结构，新增 use_cpca 开关；CPCA 可放任意 stage，默认 layer3+4）
# =============================================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv2d(in_channel, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, groups=groups, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channel*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True,
                 groups=1, width_per_group=64,
                 use_cpca=True, cpca_positions=("3", "4")):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # ---- 即插即用：CPCA（任意 stage 出口，默认 layer3+4）----
        self.use_cpca = use_cpca
        cpca_channels = {"stem": 64 * block.expansion, "1": 64 * block.expansion,
                         "2": 128 * block.expansion, "3": 256 * block.expansion, "4": 512 * block.expansion}
        self.cpca = nn.ModuleDict()
        for p in ("stem", "1", "2", "3", "4"):
            self.cpca[p] = CPCAResidual(cpca_channels[p]) if (use_cpca and p in cpca_positions) else nn.Identity()
        # ----------------------------------------------------------

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # gamma 是 Parameter，保持 0 → 初始等价于原始 ResNet。

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = [block(self.in_channel, channel, downsample=downsample, stride=stride,
                        groups=self.groups, width_per_group=self.width_per_group)]
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel,
                                groups=self.groups, width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.cpca["stem"](x)
        x = self.layer1(x); x = self.cpca["1"](x)
        x = self.layer2(x); x = self.cpca["2"](x)
        x = self.layer3(x); x = self.cpca["3"](x)
        x = self.layer4(x); x = self.cpca["4"](x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


def resnet34(num_classes=1000, include_top=True, use_cpca=True, cpca_positions=("3", "4")):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top,
                  use_cpca=use_cpca, cpca_positions=cpca_positions)


def resnet34_base(num_classes=1000, include_top=True):
    """基线臂：与原始 ResNet34 逐字节一致。"""
    return resnet34(num_classes=num_classes, include_top=include_top, use_cpca=False)


def resnet34_cpca(num_classes=1000, include_top=True):
    """改进臂：layer3+layer4 出口插入原始 CPCA（零初始化门控残差）。"""
    return resnet34(num_classes=num_classes, include_top=include_top, use_cpca=True)


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    base = resnet34_base(num_classes=3)
    cpca = resnet34_cpca(num_classes=3)
    print("base out:", tuple(base(x).shape), "| params %.3fM" % (sum(p.numel() for p in base.parameters())/1e6))
    print("cpca out:", tuple(cpca(x).shape), "| params %.3fM" % (sum(p.numel() for p in cpca.parameters())/1e6))