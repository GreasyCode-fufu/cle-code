import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
#  本文件 = 原 resnet34_degconv_cpca.py，但【仅升级 DEGConv 为改进版】，CPCA 原样保留。
#  DEGConv 改进：可微软分 bin + 空间方向嵌入(上采样) + 去掉 2x2 patch 接缝 + 真实空间 cell；
#  升级后不再有尺寸限制，可放任意 stage（默认仍 layer1+2，便于与旧实验对比）。
#  CPCA：与所给代码完全一致（共享 1x1 卷积、固定 7/11/21 核），未做任何改动。
# =============================================================================

# ============================ DEGConv（改进版）============================
class DirectionalStripEdgeConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.channel_reduction = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=bias)
        self.horizontal_strip_conv = nn.Conv2d(hidden_channels, hidden_channels,
                                               kernel_size=(1, kernel_size), stride=1,
                                               padding=(0, kernel_size // 2), groups=hidden_channels)
        self.vertical_strip_conv = nn.Conv2d(hidden_channels, hidden_channels,
                                             kernel_size=(kernel_size, 1), stride=1,
                                             padding=(kernel_size // 2, 0), groups=hidden_channels)
        self.channel_fusion = nn.Conv2d(hidden_channels * 2, out_channels, kernel_size=1, bias=True)

    def forward(self, feature_map):
        reduced = self.channel_reduction(feature_map)
        h = self.horizontal_strip_conv(reduced)
        v = self.vertical_strip_conv(reduced)
        return self.channel_fusion(torch.cat([h, v], dim=1))


class HoGDirectionGuidedEdgeGateConvV2(nn.Module):
    def __init__(self, channels, num_orientation_bins=8, cell_grid=8, soft_bin_sigma_scale=1.0):
        super().__init__()
        self.num_orientation_bins = num_orientation_bins
        self.cell_grid = cell_grid
        self.soft_bin_sigma_scale = soft_bin_sigma_scale

        # 方向嵌入编码器：不再有全局池化，输出保留空间网格
        self.direction_embedding_encoder = nn.Sequential(
            nn.Conv2d(num_orientation_bins, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(channels // 8, channels),
            nn.ReLU(inplace=True))
        self.edge_gate_generator = nn.Sequential(
            DirectionalStripEdgeConv(channels, channels // 2, channels),
            nn.GroupNorm(channels // 8, channels))
        self.feature_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.GroupNorm(channels // 8, channels))
        self.post_edge_fusion = nn.Sequential(
            DirectionalStripEdgeConv(channels, channels // 2, channels, kernel_size=3),
            nn.GroupNorm(channels // 8, channels))
        self.gate_activation = nn.Sigmoid()

        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                                     dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                                     dtype=torch.float32).view(1, 1, 3, 3))
        bin_center_start = math.pi / (2 * num_orientation_bins)
        centers = torch.linspace(bin_center_start, math.pi - bin_center_start, num_orientation_bins)
        self.register_buffer('bin_centers', centers.view(1, num_orientation_bins, 1, 1))

    def compute_hog_direction_prior(self, feature):
        """可微软分 bin + 自适应真实 cell 的方向先验，输出 [B, num_bins, gh, gw]。"""
        gray = feature.mean(dim=1, keepdim=True)
        b, _, h, w = gray.shape
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        angle = torch.abs(torch.atan2(gy, gx))                        # [B,1,H,W] in [0,pi]

        bin_width = math.pi / self.num_orientation_bins
        sigma = bin_width * self.soft_bin_sigma_scale
        # 高斯软分配：每个像素角度对各 bin 中心的软隶属度（可微）
        diff = angle - self.bin_centers                               # [B,num_bins,H,W]
        soft = torch.exp(-0.5 * (diff / sigma) ** 2)
        soft = soft / (soft.sum(dim=1, keepdim=True) + 1e-6)          # 每像素方向分布归一

        # 自适应聚合到真实空间 cell 网格（覆盖整图、不裁边界、适配任意尺寸）
        gh = min(self.cell_grid, h)
        gw = min(self.cell_grid, w)
        cell_hist = F.adaptive_avg_pool2d(soft, (gh, gw))             # [B,num_bins,gh,gw]
        # 按 bin 中心角加权（保留原版"方向先验"语义）
        prior = cell_hist * self.bin_centers
        return prior

    def forward(self, input_feature):
        residual_feature = input_feature
        b, c, h, w = input_feature.shape

        prior = self.compute_hog_direction_prior(input_feature)       # [B,num_bins,gh,gw]
        direction_embedding = self.direction_embedding_encoder(prior)  # [B,C,gh,gw]
        # 改进①：上采样回特征图尺寸，得到空间方向嵌入（而非全局向量广播）
        direction_embedding = F.interpolate(direction_embedding, size=(h, w),
                                            mode='bilinear', align_corners=False)

        gate_weight = self.gate_activation(
            self.edge_gate_generator(input_feature + direction_embedding))
        projected_feature = self.feature_projection(input_feature)
        gated_edge_feature = gate_weight * projected_feature

        restored_feature = gated_edge_feature + residual_feature
        return self.post_edge_fusion(restored_feature)


class DEGConvResidual(nn.Module):
    """即插即用封装：零初始化 LayerScale 门控残差。gamma=0 时初始 output==input。"""
    def __init__(self, channels: int, num_orientation_bins: int = 8) -> None:
        super().__init__()
        self.degconv = HoGDirectionGuidedEdgeGateConvV2(channels, num_orientation_bins=num_orientation_bins)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.degconv(x)

# =============================================================================
#  CPCA —— Channel Prior Convolutional Attention（替换原 CBAM 空间注意力）
#  通道注意力(avg+max) + 多尺度条带深度卷积空间注意力。任意尺寸可跑。
#  注: 此版 self.conv 在 forward 中复用三次（与所给代码一致）；若想用三个独立
#      卷积，把下方 self.conv 拆成 self.conv_gp / self.conv_sa / self.conv_out。
# =============================================================================

class CPCAChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(CPCAChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, stride=1, bias=True)
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
        self.ca = CPCAChannelAttention(input_channels=channels,
                                       internal_neurons=channels // channelAttention_reduce)
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
    """即插即用封装：零初始化门控残差。gamma=0 时初始 output==input（保护预训练）。"""
    def __init__(self, channels: int, channelAttention_reduce: int = 4) -> None:
        super().__init__()
        self.cpca = CPCA(channels, channelAttention_reduce=channelAttention_reduce)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.cpca(x)

# =============================================================================
#  ResNet：DEGConv(stem/1/2) + CPCA(任意 stage，默认 3/4)
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
                 use_degconv=True, degconv_positions=("1", "2"),
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

        # ---- DEGConv（改进版，可放任意 stage；默认 layer1+2）----
        self.use_degconv = use_degconv
        deg_channels = {"stem": 64 * block.expansion, "1": 64 * block.expansion,
                        "2": 128 * block.expansion, "3": 256 * block.expansion, "4": 512 * block.expansion}
        self.degconv = nn.ModuleDict()
        for p in ("stem", "1", "2", "3", "4"):
            self.degconv[p] = DEGConvResidual(deg_channels[p]) if (use_degconv and p in degconv_positions) else nn.Identity()

        # ---- CPCA 注意力（任意 stage）----
        self.use_cpca = use_cpca
        cpca_channels = {"stem": 64 * block.expansion, "1": 64 * block.expansion,
                         "2": 128 * block.expansion, "3": 256 * block.expansion, "4": 512 * block.expansion}
        self.cpca = nn.ModuleDict()
        for p in ("stem", "1", "2", "3", "4"):
            self.cpca[p] = CPCAResidual(cpca_channels[p]) if (use_cpca and p in cpca_positions) else nn.Identity()
        # ------------------------------------------

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
        x = self.degconv["stem"](x); x = self.cpca["stem"](x)
        x = self.layer1(x); x = self.degconv["1"](x); x = self.cpca["1"](x)
        x = self.layer2(x); x = self.degconv["2"](x); x = self.cpca["2"](x)
        x = self.layer3(x); x = self.degconv["3"](x); x = self.cpca["3"](x)
        x = self.layer4(x); x = self.degconv["4"](x); x = self.cpca["4"](x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


def resnet34(num_classes=1000, include_top=True,
             use_degconv=True, degconv_positions=("1", "2"),
             use_cpca=True, cpca_positions=("3", "4")):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top,
                  use_degconv=use_degconv, degconv_positions=degconv_positions,
                  use_cpca=use_cpca, cpca_positions=cpca_positions)


def resnet34_base(num_classes=1000, include_top=True):
    """基线臂：与原始 ResNet34 逐字节一致。"""
    return resnet34(num_classes=num_classes, include_top=include_top, use_degconv=False, use_cpca=False)


def resnet34_degconv(num_classes=1000, include_top=True):
    """仅 DEGConv：layer1+layer2。"""
    return resnet34(num_classes=num_classes, include_top=include_top, use_degconv=True, use_cpca=False)


def resnet34_cpca(num_classes=1000, include_top=True):
    """仅 CPCA：layer3+layer4。"""
    return resnet34(num_classes=num_classes, include_top=include_top, use_degconv=False, use_cpca=True)


def resnet34_degconv_cpca(num_classes=1000, include_top=True):
    """组合臂：DEGConv 浅层(layer1+2) + CPCA 深层(layer3+4)。"""
    return resnet34(num_classes=num_classes, include_top=include_top,
                    use_degconv=True, degconv_positions=("1", "2"),
                    use_cpca=True, cpca_positions=("3", "4"))


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    base = resnet34_base(num_classes=3)
    combo = resnet34_degconv_cpca(num_classes=3)
    print("base  out:", tuple(base(x).shape), "| params %.3fM" % (sum(p.numel() for p in base.parameters())/1e6))
    combo.train(); print("degconv+cpca train out:", tuple(combo(x).shape))
    combo.eval();  print("degconv+cpca eval  out:", tuple(combo(x).shape), "| params %.3fM" % (sum(p.numel() for p in combo.parameters())/1e6))