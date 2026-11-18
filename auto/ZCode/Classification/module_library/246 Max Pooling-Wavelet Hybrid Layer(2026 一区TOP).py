import torch
from torch import nn
import pywt
from torch.autograd import Function

"""
    论文地址：https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11352975
    论文题目：PQGNet: Perceptual Query Guided Network for Infrared Small Target Detection(2026 一区TOP)
    中文题目：基于频率结构解耦与区域感知融合的微光图像超分辨率双路径学习 (ICME 2026)
    讲解视频：https://www.bilibili.com/video/BV1Dtey6hEqS/
    最大池化-小波混合层（Max Pooling-Wavelet Hybrid Layer，MWHL）
        实际意义：①红外小目标本身像素占比极小、响应弱，经过多层下采样后，目标特征可能在低频信息中被不断削弱，甚至完全消失。本论文指出，U型网络多次下采样会使目标特征在小波低频分量中逐渐减弱。②单独使用小波下采样存在低频退化问题：如果DWT下采样只保留低频分量，虽然能保留整体结构，但会丢弃部分与小目标相关的高频细节和边缘信息。对于小目标，这种损失会进一步加重小目标的消失和后续边缘重建困难。
        核心思想：通过融合最大池化的局部强响应与小波变换的低频结构信息，实现互补式下采样，增强小目标特征保留能力。
        代码流程：输入特征 → DWT提取LL低频分量 + MaxPool提取强响应 → 通道拼接 → 1×1卷积融合降维 → 残差块增强 → 输出特征。
"""


# 带LeakyReLU的残差卷积块
class ResidualLeakBlock(nn.Module):
    # 初始化：定义卷积、归一化、激活函数
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 输入通道转换层
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        # 残差主干：5x5 + 3x3卷积
        self.body = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        # 激活函数
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    # 前向传播
    def forward(self, x):
        # 通道转换
        x = self.proj(x)
        # 保留残差
        residual = x
        # 主干卷积
        x = self.body(x)
        # 残差相加后激活
        out = self.relu(x + residual)
        return out

# 可微小波变换前向传播函数
class DWT_Function(Function):
    # 前向计算
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        # 连续化张量
        x = x.contiguous()
        # 保存参数用于反向传播
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]
        # 4种小波滤波
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        # 4个分量拼接
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    # 反向传播
    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            # 转置卷积还原梯度
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)
        return dx, None, None, None, None

# 可微逆小波变换函数
class IDWT_Function(Function):
    # 前向计算
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape
        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        # 逆小波变换（上采样）
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    # 反向传播
    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

# 逆小波变换模块封装
class IDWT_2D(nn.Module):
    # 初始化小波滤波器
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        # 构造4个二维重构滤波器
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
        # 增加维度
        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        # 注册为模型参数
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    # 前向传播
    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

# 小波变换模块封装
class DWT_2D(nn.Module):
    # 初始化小波滤波器
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        # 构造4个二维分解滤波器
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
        # 注册参数
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))
        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    # 前向传播
    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

# 最大池化-小波混合下采样层（核心模块）
class MWHL(nn.Module):
    # 初始化双分支下采样结构
    def __init__(self, channel=32, wave = 'haar'):
        super().__init__()
        # 小波变换分支
        self.dwt = DWT_2D(wave)
        self.channel = channel
        # 最大池化分支
        self.maxpool = nn.MaxPool2d(2,2)
        # 1x1卷积融合双分支
        self.conv_D = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0)
        # 残差增强
        self.ResBlock = ResidualLeakBlock(channel, channel)

    # 前向传播
    def forward(self, x):
        # 小波分解得到4个子带
        e_dwt = self.dwt(x)
        # 拆分出低频分量
        e_ll, e_lh, e_hl, e_hh = e_dwt.split(self.channel, 1)
        # 最大池化下采样
        e_down = self.maxpool(x)
        # 拼接并降维
        e2 = self.conv_D(torch.cat([e_ll, e_down], dim=1))
        # 残差增强输出
        out = self.ResBlock(e2)
        return out

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = MWHL(32)
    y = model(x)
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")