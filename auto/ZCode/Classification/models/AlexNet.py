import math
import torch.nn.functional as F
import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = SE_Block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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

# class SE(nn.Module):
#     def __init__(self,in_channerls, se_ratio=0.25, bias=True, act_fn='mish', static_padding=False, image_size=None):
#         super().__init__()
#         squeezed_channels = max(int(in_channerls * se_ratio), 1)
#         if static_padding:
#             self.reduce = Conv2dStaticSamePadding(in_channerls, squeezed_channels, 1, image_size=image_size, bias=bias)
#
# class NonLocal(nn.Module):
#     def __init__(self, channel):
#         super(NonLocalBlock, self).__init__()
#         self.inter_channel = channel // 2
#         self.conv_phi = nn.Conv2d(channel, self.inter_channel, 1, 1,0, False)
#         self.conv_theta = nn.Conv2d(channel, self.inter_channel, 1, 1,0, False)
#         self.conv_g = nn.Conv2d(channel, self.inter_channel, 1, 1, 0, False)
#         self.softmax = nn.Softmax(dim=1)
#         self.conv_mask = nn.Conv2d(self.inter_channel, channel, 1, 1, 0, False)
#     def forward(self, x):
#         # [N, C, H , W]
#         b, c, h, w = x.size()
#         # 获取phi特征，维度为[N, C/2, H * W]，注意是要保留batch和通道维度的，是在HW上进行的
#         x_phi = self.conv_phi(x).view(b, c, -1)
#         # 获取theta特征，维度为[N, H * W, C/2]
#         x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()        # 获取g特征，维度为[N, H * W, C/2]
#         x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
#         # 对phi和theta进行矩阵乘，[N, H * W, H * W]
#         mul_theta_phi = torch.matmul(x_theta, x_phi)
#         # softmax拉到0~1之间
#         mul_theta_phi = self.softmax(mul_theta_phi)
#         # 与g特征进行矩阵乘运算，[N, H * W, C/2]
#         mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
#         # [N, C/2, H, W]
#         mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
#         # 1X1卷积扩充通道数
#         mask = self.conv_mask(mul_theta_phi_g)
#         out = mask + x # 残差连接
#         return out
