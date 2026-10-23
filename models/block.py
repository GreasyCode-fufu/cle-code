import math
import torch
import torch.nn as nn
import torch.nn.functional as F





class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

class EFG_Module(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(EFG_Module, self).__init__()

        out_channels = in_channels // reduction

        self.conv_efg = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=False),  # 3x3
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(2, in_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # self.conv_1X1 = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # 1x1
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        # )

        self.conv_revert = nn.Sequential(
            nn.Conv2d(out_channels*2, in_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = x
        x = self.conv_efg(x)
        # x1 = self.conv_1X1(x1)
        x = torch.cat((x, x1), dim=1)
        x = self.conv_revert(x)
        return x


class LIG(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(LIG, self).__init__()

        out_channels = in_channels // reduction

        self.conv_lig = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        shortcut = x
        x = self.conv_lig(x)
        x = shortcut + x
        return x

class PVT_EFG_Module(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(PVT_EFG_Module, self).__init__()

        out_channels = in_channels // reduction

        self.conv_efg = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=False),  # 3x3
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(2, in_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_1X1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_revert = nn.Sequential(
            nn.Conv2d(576, in_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = x
        x1 = self.conv_efg(x1)
        x1 = self.conv_1X1(x1)
        # x = self.conv_1X1(x)
        x = torch.cat((x, x1), dim=1)
        x = self.conv_revert(x)
        return x



class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)
        self.primary_conv = nn.Sequential(nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                                          nn.BatchNorm2d(init_channels),
                                          nn.ReLU(inplace=True) if relu else nn.Sequential(), )
        # cheap操作，注意利用了分组卷积进行通道分离
        self.cheap_operation = nn.Sequential(nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                                             nn.BatchNorm2d(new_channels),
                                             nn.ReLU(inplace=True) if relu else nn.Sequential(),)
    def forward(self, x):
        x1 = self.primary_conv(x)
        #主要的卷积操作
        x2 = self.cheap_operation(x1) # cheap变换操作
        out = torch.cat([x1,x2], dim=1) # 二者cat到一起
        return out[:,:self.oup,:,:]


class ASPP(nn.Module):
    def __init__(self, in_channel=48, depth=48):
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        # 不同空洞率的卷积
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
    def forward(self, x):
        size = x.shape[2:]     # 池化分支
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
        # 不同空洞率的卷积
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        # 汇合所有尺度的特征
        x = torch.cat([image_features, atrous_block1, atrous_block6,atrous_block12, atrous_block18], dim=1)
        # 利用1X1卷积融合特征输出
        x = self.conv_1x1_output(x)
        return x

class channel_attention(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super().__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        max_pool = self.max_pool(inputs)
        avg_pool = self.avg_pool(inputs)
        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        x = x_maxpool + x_avgpool
        x = self.sigmoid(x)
        x = x.view([b, c, 1, 1])

        outputs = inputs * x

        return outputs

class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)

        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        x = self.conv(x)
        x = self.sigmoid(x)

        outputs = inputs * x

        return outputs


# （3）CBAM注意力机制
class CBAM(nn.Module):
    # 初始化，in_channel和ratio=4代表通道注意力机制的输入通道数和第一个全连接下降的通道数
    # kernel_size代表空间注意力机制的卷积核大小
    def __init__(self, in_channel, ratio=4, kernel_size=7):
        super().__init__()
        # 实例化通道注意力机制
        self.channel_attention = channel_attention(in_channel=in_channel, ratio=ratio)
        # 实例化空间注意力机制
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)

    # 前向传播
    def forward(self, inputs):
        # 先将输入图像经过通道注意力机制
        x = self.channel_attention(inputs)

        # 然后经过空间注意力机制
        x = self.spatial_attention(x)

        return x