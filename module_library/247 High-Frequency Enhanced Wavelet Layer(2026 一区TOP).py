import torch
from torch import nn
import pywt
from torch.autograd import Function


"""
    论文地址：https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11352975
    论文题目：PQGNet: Perceptual Query Guided Network for Infrared Small Target Detection(2026 一区TOP)
    中文题目：基于频率结构解耦与区域感知融合的微光图像超分辨率双路径学习 (ICME 2026)
    讲解视频：https://www.bilibili.com/video/BV16Nhv61EZw/
    高频增强小波层（High-Frequency Enhanced Wavelet Layer，HFWL）
        实际意义：①深层特征中的高频边缘信息丢失问题：在连续下采样过程中，如果深层主要保留低频分量，那么包含边缘/结构/细节高频信息会逐渐被丢弃，从而造成小目标边界模糊、结构信息缺失。
                ②深层语义特征难以恢复精细目标轮廓的问题：深层特征虽然具有较强的语义信息，但空间细节已经明显弱化。对于只有几个像素的红外小目标，仅依赖深层语义很难准确捕获边缘特征。
                ③高频信息中目标边缘与背景噪声混杂的问题：高频成分并不全部是有效目标信息，复杂背景和噪声同样会产生高频响应。
        核心思想：利用注意力筛选有效高频细节，并通过小波逆变换将其引入深层语义重建，以增强小目标边缘与结构信息特征能力。
        代码流程：深层语义特征 + 编码器特征 → DWT提取LH/HL/HH高频边缘信息 → 通道注意力增强深层语义特征并生成空间注意力 → 空间注意力筛选高频细节 → 深层语义特征与增强高频子带拼接 →
                            IDWT进行小波重建与上采样 → 3×3卷积进一步增强 → 与编码器特征通道拼接 → 残差块融合 → 输出重建特征。
"""

# 残差卷积块
class ResidualLeakBlock(nn.Module):
    # 初始化网络层
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 输入通道转换
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        # 残差主干卷积
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
        x = self.proj(x)
        residual = x
        x = self.body(x)
        # 残差相加后激活
        out = self.relu(x + residual)
        return out

# 可微小波变换前向函数
class DWT_Function(Function):
    # 前向计算
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        # 保存参数用于反向传播
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]
        # 4种小波滤波，步长2下采样
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        # 4个子带通道拼接
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
            # 转置卷积计算梯度
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
        # 逆小波上采样
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

# 逆小波变换封装模块
class IDWT_2D(nn.Module):
    # 初始化小波滤波器
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        # 构造4个二维重构核
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
        # 维度扩展
        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        # 注册固定参数
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    # 前向传播
    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

# 小波变换封装模块
class DWT_2D(nn.Module):
    # 初始化小波滤波器
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        # 构造4个二维分解核
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = rec_hi = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
        # 注册固定参数
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

# 交替通道拼接模块
class AlternateCat(nn.Module):
    # 初始化
    def __init__(self, dim=1, num=3):
        super().__init__()
        self.dim = dim
        self.num = num

    # 前向传播
    def forward(self, x_list):
        # 检查输入合法性
        assert len(x_list) == self.num, f'输入张量数量错误！'
        for i in range(self.num):
            assert x_list[0].shape == x_list[i].shape, f'第{i}个输入张量尺寸不匹配！'
        size = x_list[0].size(self.dim)
        # 拆分每个特征的通道
        x_list_slices = []
        for i in range(self.num):
            x_list_slices.append(torch.split(x_list[i], 1, dim=self.dim))
        # 交替拼接通道
        interleaved_slices = []
        for i in range(size):
            for j in range(self.num):
                interleaved_slices.append(x_list_slices[j][i])
        concatenated = torch.cat(interleaved_slices, dim=self.dim)
        return concatenated

# 交替通道-空间注意力模块
class AlCattention(nn.Module):
    # 初始化
    def __init__(self, dim):
        super(AlCattention, self).__init__()
        self.dim = dim
        # 全局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 交替拼接
        self.alcat = AlternateCat(dim=1, num=3)
        # 通道注意力MLP
        self.share_mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, stride=1, padding=0, bias=True, groups=self.dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        # 空间注意力卷积
        self.spconv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

    # 前向传播
    def forward(self, x, highf):
        # 1. 通道注意力
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        std = torch.std(x, dim=(2, 3), keepdim=True)
        # 交替拼接
        x_ams = self.alcat([x_avg, x_max, std])
        # 计算权重并增强
        channel_weights = self.share_mlp(x_ams)
        x_1 = x * channel_weights + x

        # 2. 空间注意力
        avg_out = torch.mean(x_1, dim=1, keepdim=True)
        max_out, _ = torch.max(x_1, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.spconv(spatial_features)

        # 3. 增强高频子带
        dwt = [x_1]
        for i in range(len(highf)):
            temp = highf[i] * spatial_weights.expand_as(highf[i]) + highf[i]
            dwt.append(temp)
        # 拼接高低频
        dwt = torch.cat(dwt, dim=1)
        return dwt

# 高频增强小波层（核心模块）
class HEWL(nn.Module):
    # 初始化
    def __init__(self, dim=32, out_ch=1, scale_factor=8, wave='haar'):
        super().__init__()
        self.dim = dim
        # 小波变换
        self.dwt = DWT_2D(wave)
        self.idwt = IDWT_2D(wave)
        # 注意力模块
        self.alca = AlCattention(dim=dim)
        # 特征增强卷积
        self.Conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(inplace=True)
        )
        # 输出预测头
        self.PredOut = nn.Sequential(
            nn.Conv2d(dim, out_ch, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=scale_factor)
        )
        # 最终残差块
        self.Out = ResidualLeakBlock(dim * 2, dim)

    # 前向传播
    def forward(self, x, e):
        # 1. 编码器特征小波分解
        e_dwt = self.dwt(e)
        e_ll, e_lh, e_hl, e_hh = e_dwt.split(self.dim, 1)
        e_high = [e_lh, e_hl, e_hh]

        # 2. 注意力增强
        x_idwt = self.alca(x, e_high)

        # 3. 逆小波重建
        x = self.idwt(x_idwt)

        # 4. 特征增强 + 输出预测
        d = self.Conv(x)
        # 5. 融合残差输出
        d = torch.cat((e, d), dim=1)
        out = self.Out(d)
        return out

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 64, 16, 16).to(device)
    e = torch.randn(1, 64, 32, 32).to(device)
    model = HEWL(64).to(device)
    y = model(x, e)
    print("输入特征维度：", x.shape)
    print("输入特征维度：", e.shape)
    print("输出特征维度：", y.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")