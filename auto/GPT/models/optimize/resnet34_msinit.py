import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops

# =============================================================================
#  DEGConv —— Direction-guided Edge Gated Convolution（方向引导边缘门控卷积）
#  来源: 去噪论文 (CVPR 2026) 的 HoGDirectionGuidedEdgeGateConv。
#  此处用途: 即插即用的"HoG 方向先验 + 边缘门控"增强块。
#
#  对原始代码做了两处必要修正（否则在 ResNet 特征图上无法运行）：
#   1) HoG 的逐 cell Python 循环 + torch.bincount 改为等价的向量化 one-hot 统计
#      （数值逐 bit 一致，已验证；避免 GPU 上每个 cell 一次同步导致训练极慢）。
#   2) cropped_angle 的 .view 改为 .reshape（裁剪后张量非连续，原 .view 会直接报错）。
#
#  限制: 内部按 2x2 patch 切分(要求 H、W 为偶数)，再按 8x8 cell 统计方向，
#        故只能放在 stem / layer1 / layer2（56/56/28）；layer3(14→cell数0) 与
#        layer4(7,奇数) 都会崩，已在构建时禁止。
# =============================================================================

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


def split_image_to_local_patches(feature_map):
    return einops.rearrange(feature_map, 'b c (num_h h) (num_w w) -> (num_h num_w b) c h w',
                            num_h=2, num_w=2)


def merge_local_patches_to_image(patch_feature):
    return einops.rearrange(patch_feature, '(num_h num_w b) c h w -> b c (num_h h) (num_w w)',
                            num_h=2, num_w=2)


class HoGDirectionGuidedEdgeGateConv(nn.Module):
    def __init__(self, channels, num_orientation_bins=8, cell_size=(8, 8)):
        super().__init__()
        self.num_orientation_bins = num_orientation_bins
        self.cell_size = cell_size
        self.direction_embedding_encoder = nn.Sequential(
            nn.Conv2d(num_orientation_bins, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(channels // 8, channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))
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
        # 预存 Sobel 核为 buffer（随 .to(device) 迁移）
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                                     dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                                     dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, input_feature):
        residual_feature = input_feature
        local_patch_features = split_image_to_local_patches(input_feature)
        hog_direction_prior = self.compute_hog_direction_prior(local_patch_features)
        direction_embedding = self.direction_embedding_encoder(hog_direction_prior)
        gate_weight = self.gate_activation(
            self.edge_gate_generator(local_patch_features + direction_embedding))
        projected_feature = self.feature_projection(local_patch_features)
        gated_edge_feature = gate_weight * projected_feature
        restored_feature = merge_local_patches_to_image(gated_edge_feature)
        restored_feature = restored_feature + residual_feature
        return self.post_edge_fusion(restored_feature)

    def compute_hog_direction_prior(self, patch_feature):
        """向量化 HoG 方向先验（与原版逐 cell bincount 逐 bit 等价）。"""
        grayscale_feature = patch_feature.mean(dim=1, keepdim=True)
        batch_size, _, height, width = grayscale_feature.shape
        gradient_x = F.conv2d(grayscale_feature.float(), self.sobel_x, padding=1)
        gradient_y = F.conv2d(grayscale_feature.float(), self.sobel_y, padding=1)
        gradient_angle = torch.abs(torch.atan2(gradient_y, gradient_x))

        cell_height, cell_width = self.cell_size
        num_cells_h = int(height / cell_height)
        num_cells_w = int(width / cell_width)
        cropped_angle = gradient_angle[:, :, :num_cells_h * cell_height, :num_cells_w * cell_width]
        # reshape（原版用 view，裁剪后非连续会报错）；分组语义与原版一致
        cell_angle_values = cropped_angle.reshape(batch_size, num_cells_h, num_cells_w, -1)

        bin_width = math.pi / self.num_orientation_bins
        orientation_bin_indices = torch.clamp(
            (cell_angle_values / bin_width).floor().long(), 0, self.num_orientation_bins - 1)
        # one-hot 求和 == 每个 cell 的 bincount（向量化，无 Python 循环、无 GPU 同步）
        orientation_histograms = F.one_hot(
            orientation_bin_indices, self.num_orientation_bins).sum(dim=-2).float()
        orientation_histograms = orientation_histograms / (cell_height * cell_width)

        bin_center_start = math.pi / (2 * self.num_orientation_bins)
        orientation_bin_centers = torch.linspace(
            bin_center_start, math.pi - bin_center_start, self.num_orientation_bins,
            device=patch_feature.device)
        orientation_bin_centers = orientation_bin_centers.repeat(
            batch_size, num_cells_h, num_cells_w, 1)
        hog_direction_prior = orientation_bin_centers * orientation_histograms
        return hog_direction_prior.permute(0, 3, 1, 2)


class DEGConvResidual(nn.Module):
    """即插即用封装：零初始化 LayerScale 门控残差。gamma=0 时初始 output==input。"""
    def __init__(self, channels: int, num_orientation_bins: int = 8) -> None:
        super().__init__()
        self.degconv = HoGDirectionGuidedEdgeGateConv(channels, num_orientation_bins=num_orientation_bins)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.degconv(x)




class SpatialAttention(nn.Module):
    """CBAM 式空间注意力：通道维 avg+max 池化 -> 7x7 卷积 -> sigmoid，输出 [B,1,H,W] 注意力图。"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return attn


class SpatialAttentionResidual(nn.Module):
    """即插即用封装：零初始化门控的空间注意力。
    out = x + gamma*(attn-1)*x，gamma=0 时初始 output==input（保护预训练），
    gamma 学到 1 时等价于标准空间注意力 x*attn。"""
    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.sa = SpatialAttention(kernel_size)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.sa(x)
        return x + self.gamma * (attn - 1.0) * x


# =============================================================================
#  ResNet（原始结构，新增 use_degconv 开关；DEGConv 仅允许 stem/layer1/layer2）
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
                 groups=1, width_per_group=64, use_degconv=True, degconv_positions=("1", "2"),
                 use_sa=True, sa_positions=("3", "4")):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group

        allowed = {"stem", "1", "2"}
        bad = [p for p in degconv_positions if p not in allowed]
        if use_degconv and bad:
            raise ValueError(f"DEGConv 仅支持位置 {sorted(allowed)}（layer3/4 特征图过小会崩）；非法位置: {bad}")

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # ---- 即插即用：DEGConv（仅 stem/layer1/layer2）----
        self.use_degconv = use_degconv
        pos_channels = {"stem": 64 * block.expansion, "1": 64 * block.expansion, "2": 128 * block.expansion}
        self.degconv = nn.ModuleDict()
        for p in ("stem", "1", "2"):
            if use_degconv and p in degconv_positions:
                self.degconv[p] = DEGConvResidual(pos_channels[p])
            else:
                self.degconv[p] = nn.Identity()
        # ---------------------------------------------------

        # ---- 即插即用：空间注意力（任意位置 stem/1/2/3/4）----
        self.use_sa = use_sa
        sa_pos_channels = {"stem": 64 * block.expansion, "1": 64 * block.expansion,
                           "2": 128 * block.expansion, "3": 256 * block.expansion,
                           "4": 512 * block.expansion}
        self.sa = nn.ModuleDict()
        for p in ("stem", "1", "2", "3", "4"):
            if use_sa and p in sa_positions:
                self.sa[p] = SpatialAttentionResidual(sa_pos_channels[p])
            else:
                self.sa[p] = nn.Identity()
        # ---------------------------------------------------

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
        x = self.degconv["stem"](x); x = self.sa["stem"](x)
        x = self.layer1(x); x = self.degconv["1"](x); x = self.sa["1"](x)
        x = self.layer2(x); x = self.degconv["2"](x); x = self.sa["2"](x)
        x = self.layer3(x); x = self.sa["3"](x)
        x = self.layer4(x); x = self.sa["4"](x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


def resnet34(num_classes=1000, include_top=True, use_degconv=True, degconv_positions=("1", "2"),
             use_sa=True, sa_positions=("3", "4")):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top,
                  use_degconv=use_degconv, degconv_positions=degconv_positions,
                  use_sa=use_sa, sa_positions=sa_positions)


def resnet34_base(num_classes=1000, include_top=True):
    """基线臂：与原始 ResNet34 逐字节一致。"""
    return resnet34(num_classes=num_classes, include_top=include_top, use_degconv=False, use_sa=False)


def resnet34_degconv(num_classes=1000, include_top=True):
    """仅 DEGConv：layer1+layer2（无空间注意力）。"""
    return resnet34(num_classes=num_classes, include_top=include_top,
                    use_degconv=True, use_sa=False)


def resnet34_sa(num_classes=1000, include_top=True):
    """仅空间注意力：layer3+layer4（无 DEGConv），用于消融。"""
    return resnet34(num_classes=num_classes, include_top=include_top,
                    use_degconv=False, use_sa=True)


def resnet34_degconv_sa(num_classes=1000, include_top=True):
    """组合臂：DEGConv 放浅层(layer1+2) + 空间注意力放深层(layer3+4)。"""
    return resnet34(num_classes=num_classes, include_top=include_top,
                    use_degconv=True, degconv_positions=("1", "2"),
                    use_sa=True, sa_positions=("3", "4"))


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    base = resnet34_base(num_classes=3)
    combo = resnet34_degconv_sa(num_classes=3)
    print("base  out:", tuple(base(x).shape), "| params %.3fM" % (sum(p.numel() for p in base.parameters())/1e6))
    combo.train(); print("degconv+sa train out:", tuple(combo(x).shape))
    combo.eval();  print("degconv+sa eval  out:", tuple(combo(x).shape), "| params %.3fM" % (sum(p.numel() for p in combo.parameters())/1e6))