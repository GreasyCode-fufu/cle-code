import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedCPCAChannelAttention(nn.Module):
    """
    Improved CPCA Channel Attention
    改进点：
    1. AvgPool 分支：建模全局平均响应
    2. MaxPool 分支：建模显著强响应
    3. StdPool 分支：建模通道内对比度/纹理变化，提高光照不均场景下的鲁棒性
    """

    def __init__(self, input_channels, internal_neurons):
        super().__init__()

        internal_neurons = max(1, internal_neurons)

        self.fc1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=internal_neurons,
            kernel_size=1,
            stride=1,
            bias=True
        )

        self.fc2 = nn.Conv2d(
            in_channels=internal_neurons,
            out_channels=input_channels,
            kernel_size=1,
            stride=1,
            bias=True
        )

    def forward(self, inputs):
        # inputs: [B, C, H, W]

        # 1. 全局平均池化分支
        x_avg = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x_avg = self.fc1(x_avg)
        x_avg = F.relu(x_avg, inplace=True)
        x_avg = self.fc2(x_avg)

        # 2. 全局最大池化分支
        x_max = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x_max = self.fc1(x_max)
        x_max = F.relu(x_max, inplace=True)
        x_max = self.fc2(x_max)

        # 3. 标准差池化分支
        # 用于增强光照不均、阴影、局部过曝情况下的鲁棒性
        x_std = torch.std(inputs, dim=(2, 3), keepdim=True, unbiased=False)
        x_std = self.fc1(x_std)
        x_std = F.relu(x_std, inplace=True)
        x_std = self.fc2(x_std)

        # 三个分支融合
        att = torch.sigmoid(x_avg + x_max + x_std)

        # 通道注意力加权
        out = inputs * att

        return out


class CPCA1(nn.Module):
    """
    ImprovedCPCA 第一版：
    StdPool通道注意力 + 自适应多尺度融合 + Sigmoid残差门控

    输入:  [B, C, H, W]
    输出:  [B, C, H, W]
    """

    def __init__(self, channels, channelAttention_reduce=4):
        super().__init__()

        internal_neurons = max(1, channels // channelAttention_reduce)

        # 改进通道注意力：AvgPool + MaxPool + StdPool
        self.ca = ImprovedCPCAChannelAttention(
            input_channels=channels,
            internal_neurons=internal_neurons
        )

        # 输入通道预处理
        self.pre_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        self.act = nn.GELU()

        # 5×5 depthwise convolution
        self.dconv5_5 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=channels,
            bias=True
        )

        # 7×7 分支：1×7 + 7×1
        self.dconv1_7 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 7),
            stride=1,
            padding=(0, 3),
            groups=channels,
            bias=True
        )

        self.dconv7_1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(7, 1),
            stride=1,
            padding=(3, 0),
            groups=channels,
            bias=True
        )

        # 11×11 分支：1×11 + 11×1
        self.dconv1_11 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 11),
            stride=1,
            padding=(0, 5),
            groups=channels,
            bias=True
        )

        self.dconv11_1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(11, 1),
            stride=1,
            padding=(5, 0),
            groups=channels,
            bias=True
        )

        # 21×21 分支：1×21 + 21×1
        self.dconv1_21 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 21),
            stride=1,
            padding=(0, 10),
            groups=channels,
            bias=True
        )

        self.dconv21_1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(21, 1),
            stride=1,
            padding=(10, 0),
            groups=channels,
            bias=True
        )

        # 自适应多尺度融合权重
        # 对应四个分支：
        # 0: x_init
        # 1: x_1, 7×7分支
        # 2: x_2, 11×11分支
        # 3: x_3, 21×21分支
        self.scale_weight = nn.Parameter(torch.ones(4))

        # 生成空间注意力
        self.spatial_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        # 输出通道融合
        self.post_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, inputs):
        # inputs: [B, C, H, W]

        identity = inputs

        # 1. 输入特征预处理
        x = self.pre_conv(inputs)
        x = self.act(x)

        # 2. StdPool 通道注意力
        x = self.ca(x)

        # 3. 基础 5×5 depthwise convolution
        x_init = self.dconv5_5(x)

        # 4. 7×7 多尺度分支
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)

        # 5. 11×11 多尺度分支
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)

        # 6. 21×21 多尺度分支
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)

        # 7. 自适应多尺度融合
        weight = torch.softmax(self.scale_weight, dim=0)

        x_ms = (
            weight[0] * x_init +
            weight[1] * x_1 +
            weight[2] * x_2 +
            weight[3] * x_3
        )

        # 8. Sigmoid 空间注意力门控
        spatial_att = torch.sigmoid(self.spatial_conv(x_ms))

        # 9. 残差门控增强
        # spatial_att 范围是 0~1
        # 1 + spatial_att 范围是 1~2
        # 表示在保留原特征的基础上增强重要区域
        out = x * (1.0 + spatial_att)

        # 10. 输出通道融合
        out = self.post_conv(out)

        # 11. 残差连接，增强训练稳定性
        out = out + identity

        return out
