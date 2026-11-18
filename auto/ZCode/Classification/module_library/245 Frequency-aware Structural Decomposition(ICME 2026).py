import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    论文地址：https://arxiv.org/pdf/2603.27301
    论文题目：Dual-Path Learning based on Frequency Structural Decoupling and Regional-Aware Fusion for Low-Light Image Super-Resolution (ICME 2026)
    中文题目：基于频率结构解耦与区域感知融合的微光图像超分辨率双路径学习 (ICME 2026)
    讲解视频：https://www.bilibili.com/video/BV1aTtM6mEZB/
    频率感知结构分解模块（Frequency-aware Structural Decomposition，FSD）
        实际意义：①亮度与纹理语义纠缠:传统方法在统一空间域中同时处理亮度和纹理，这使得模型很难区分“低照度的亮度伪影”和“真实高频纹理”，容易产生结构失真、伪纹理和感知不一致。
                ②缺乏不同频率成分的结构化建模：整体亮度主要集中在低频，而边缘、轮廓和细节主要位于高频。如果不进行分解，而是在一个统一特征空间中处理，容易产生跨频率干扰。
        实现方式：通过可学习小波分解与频率注意力自适应调整高低频信息，利用频率分解将全局结构与局部细节显式解耦出来。
"""

# 多尺度小波变换网络类
class MultiScaleWaveletNet(nn.Module):
    # 初始化函数：定义网络结构和参数
    def __init__(self, in_channels: int, out_channels: int, levels: int = 3) -> None:
        # 调用父类初始化
        super().__init__()
        # 小波分解的尺度数量
        self.levels = levels
        # 输入特征的通道数
        self.in_channels = in_channels
        # 小波分解后每个通道会变成4个频带，总通道数×4
        self.wavelet_channels = in_channels * 4
        # 每个尺度分配的输出通道数
        self.out_channels_per_level = out_channels // levels
        # 为每个尺度构建小波特征卷积层（容器）
        self.wavelet_convs = nn.ModuleList(
            [
                # 每个尺度的卷积+归一化+激活
                nn.Sequential(
                    # 二维卷积层
                    nn.Conv2d(self.wavelet_channels, self.out_channels_per_level, 3, 1, 1),
                    # 实例归一化
                    nn.InstanceNorm2d(self.out_channels_per_level),
                    # PReLU激活函数
                    nn.PReLU(),
                )
                # 循环创建levels个尺度的卷积层
                for _ in range(levels)
            ]
        )
        # 多尺度特征融合模块
        self.fusion = nn.Sequential(
            # 1×1卷积融合所有尺度特征
            nn.Conv2d(self.out_channels_per_level * levels, out_channels, 1, 1, 0),
            # 实例归一化
            nn.InstanceNorm2d(out_channels),
            # PReLU激活
            nn.PReLU(),
        )

    # 小波变换函数：对单张特征图做Haar小波分解
    def wavelet_transform(self, x: torch.Tensor) -> torch.Tensor:
        # 获取当前数据所在设备（CPU/GPU）
        device = x.device
        # 存储每个样本的小波分解结果
        coeffs = []
        # 遍历batch中的每一个样本
        for batch_index in range(x.shape[0]):
            # 取出单张图像并转为numpy格式
            image = x[batch_index, 0].detach().cpu().numpy()
            # 执行二维Haar小波一级分解
            ll, (lh, hl, hh) = pywt.dwt2(image, "haar")
            # 存储4个频带
            bands = []
            # 遍历4个小波频带
            for band in (ll, lh, hl, hh):
                # 转回张量并增加维度
                band_tensor = torch.from_numpy(band).float().unsqueeze(0).unsqueeze(0)
                # 如果尺寸不一致，上采样回原图大小
                if band_tensor.shape[2:] != x.shape[2:]:
                    band_tensor = F.interpolate(
                        band_tensor,
                        size=x.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                # 保存当前频带
                bands.append(band_tensor)
            # 4个频带在通道维度拼接
            coeffs.append(torch.cat(bands, dim=1))
        # batch维度拼接并返回原设备
        return torch.cat(coeffs, dim=0).to(device)

    # 前向传播：网络主流程
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保存所有尺度的小波特征
        multi_scale_features = []
        # 当前输入特征（会不断下采样）
        current_x = x
        # 逐层进行小波分解
        for level, wavelet_conv in enumerate(self.wavelet_convs):
            # 存储当前尺度所有通道的小波特征
            wavelet_features = []
            # 对每个通道单独做小波变换
            for channel_index in range(self.in_channels):
                # 取出单通道特征
                single_channel = current_x[:, channel_index : channel_index + 1]
                # 小波分解
                coeffs = self.wavelet_transform(single_channel)
                # 保存结果
                wavelet_features.append(coeffs)
            # 所有通道的小波特征在通道维度拼接
            wavelet_tensor = torch.cat(wavelet_features, dim=1)
            # 用卷积层提取当前尺度特征
            features = wavelet_conv(wavelet_tensor)
            # 非第一层特征上采样回输入尺寸
            if level > 0:
                features = F.interpolate(
                    features,
                    size=x.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            # 保存当前尺度特征
            multi_scale_features.append(features)
            # 未到最后一层时，对特征下采样
            if level < self.levels - 1:
                current_x = F.avg_pool2d(current_x, 2)
        # 拼接所有尺度特征并融合输出
        return self.fusion(torch.cat(multi_scale_features, dim=1))

# 频率注意力模块
class FrequencyAttention(nn.Module):
    # 初始化
    def __init__(self, channels: int) -> None:
        super().__init__()
        # 全局平均池化：输出1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全局最大池化：输出1×1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 注意力权重生成层
        self.fc = nn.Sequential(
            # 压缩通道
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            # ReLU激活
            nn.ReLU(inplace=True),
            # 恢复通道
            nn.Conv2d(channels, channels * 4, 1, bias=False),
            # Sigmoid输出0~1权重
            nn.Sigmoid(),
        )

    # 前向传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 平均池化+最大池化生成注意力权重
        attention = self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        # 特征与注意力权重相乘，增强重要频带
        return x * attention

# 频率感知结构解耦模块（主模块）
class FrequencyStructuralDecoupling(nn.Module):
    # 初始化
    def __init__(self, features: int = 64) -> None:
        super().__init__()
        # 多尺度小波网络
        self.wavelet_net = MultiScaleWaveletNet(3, features)
        # 频率注意力模块
        self.frequency_attention = FrequencyAttention(features // 4)
        # 双层卷积特征提取
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(features, features, 3, 1, 1),
            nn.BatchNorm2d(features),
            nn.PReLU(),
            nn.Conv2d(features, features, 3, 1, 1),
            nn.BatchNorm2d(features),
            nn.PReLU(),
        )
        # 结构-纹理分离层：输出6通道（3+3）
        self.separator = nn.Sequential(
            nn.Conv2d(features, 6, 3, 1, 1),
            nn.BatchNorm2d(6),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # 第一步：多尺度小波分解得到频域特征
        wavelet_features = self.wavelet_net(x)
        # 第二步：频率注意力加权
        attended_features = self.frequency_attention(wavelet_features)
        # 第三步：深度特征强化
        enhanced_features = self.feature_extraction(attended_features)
        # 第四步：分离出结构和纹理
        separated = self.separator(enhanced_features)
        # 前3通道 = 纹理分量
        texture_llr = torch.stack([separated[:, index, :, :] for index in range(3)], dim=1)
        # 后3通道 = 亮度结构分量
        luminance_llr = torch.stack([separated[:, index, :, :] for index in range(3, 6)], dim=1)
        # 返回结构、纹理、增强特征
        return luminance_llr, texture_llr, enhanced_features

if __name__ == "__main__":
    x = torch.randn(1, 32, 256, 256)
    model = FrequencyStructuralDecoupling(32)
    luminance, texture, feat = model(x)
    print("输入图像维度：", x.shape)
    print("亮度结构分量维度：", luminance.shape)
    print("纹理细节分量维度：", texture.shape)
    print("增强特征维度：", feat.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")