import torch.nn as nn
import torch.nn.functional as F

class SElayer(nn.Module):
    def __init__(self, ratio, input_aisle):
        super(SElayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_aisle, input_aisle // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_aisle // ratio, input_aisle, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        a, b, _, _ = x.size()
        # 读取批数据图片数量及通道数
        y = self.pool(x).view(a, b)
        # 经池化后输出a*b的矩阵
        y = self.fc(y).view(a, b, 1, 1)
        # 经全连接层输出【啊，吧，1，1】矩阵
        return x * y.expand_as(x)
        # 输出权重乘以特征图


class SE_block(nn.Module):
    def __init__(self, outchannels, ratio=16, ):
        super(SE_block, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=outchannels, out_channels=outchannels // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(outchannels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outchannels // 4, out_channels=outchannels // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outchannels // 4, out_channels=outchannels, kernel_size=1, stride=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            )
        self.Selayer = SElayer(ratio, outchannels)

    def forward(self, x):
        residual = x
        x = self.backbone(x)
        x = self.Selayer(x)
        x = x + residual
        return x


class SE_Idenitity(nn.Module):
    def __init__(self, inchannel, outchannels, stride=2, ratio=16, ):
        super(SE_Idenitity, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannels // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(outchannels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outchannels // 4, out_channels=outchannels // 4, kernel_size=3, stride=stride,
                      padding=1),
            nn.BatchNorm2d(outchannels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outchannels // 4, out_channels=outchannels, kernel_size=1, stride=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Conv2d(in_channels=inchannel, out_channels=outchannels, kernel_size=1, stride=stride)
        self.selayer = SElayer(ratio, outchannels)

    def forward(self, x):
        residual = self.residual(x)
        x = self.backbone(x)
        x = self.selayer(x)
        x = x + residual
        return x


class SE_Resnet(nn.Module):
    def __init__(self, layer_num, outchannel_num, num_class=10):
        super(SE_Resnet, self).__init__()
        self.initial = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=3, padding=1, stride=2))
        self.residual = self.make_layer(layer_num, outchannel_num)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_class)

    def make_layer(self, number_layer, number_outchannel):
        layer_list = []

        for i in range(len(number_layer)):
            if i == 0:
                stride = 1
            else:
                stride = 2
            layer_list.append(SE_Idenitity(number_outchannel[i], number_outchannel[i + 1], stride))
            for j in range(number_layer[i] - 1):
                layer_list.append(SE_block(number_outchannel[i + 1]))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.initial(x)
        x = self.residual(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)


def SE_Resnet18(num_classes):
    model = SE_Resnet([2, 2, 2, 2], [64, 256, 512, 1024, 2048], num_classes)
    return model

def SE_Resnet50(num_classes):
    model = SE_Resnet([3, 4, 6, 3], [64, 256, 512, 1024, 2048], num_classes)
    return model


def SE_Resnet101(num_classes):
    model = SE_Resnet([3, 4, 23, 3], [64, 256, 512, 1024, 2048], num_classes)
    return model


def SE_Resnet151(num_classes):
    model = SE_Resnet([3, 8, 36, 3], [64, 256, 512, 1024, 2048], num_classes)
    return model
