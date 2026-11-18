import torch
import torch.nn as nn

"""
    论文地址：https://arxiv.org/pdf/2603.28584v1
    论文题目：ORSIFlow: Saliency-Guided Rectified Flow for Optical Remote Sensing Salient Object Detection (ICME 2026)
    中文题目：ORSIFlow：面向光学遥感显著目标检测的显著性引导矫正流方法（ICME 2026）
    讲解视频：https://www.bilibili.com/video/BV1V1nb6oEQW/
    显著特征校准模块（Salient Feature Calibrator，SFC）
        实际意义：①显著目标边界模糊问题：由于目标区域与背景区域之间存在较低的差异，传统特征融合方法容易丢失边界信息，预测结果出现边缘断裂、分割轮廓不准确。②不同尺度目标特征表达不足问题：遥感场景中目标尺度变化非常明显。传统固定融合策略难以适应不同区域的重要性差异：大目标容易获得较强响应；小目标容易被复杂背景淹没。
        核心思想：通过双分支门控融合机制，自适应校准显著特征中的局部细节与全局语义信息，有效增强小目标响应和目标边界定位能力。
        代码流程：输入特征 → 两路1×1卷积分支映射 → 特征通道拼接 → GAP提取全局语义信息 + GMP提取局部强响应 → 动态门控权重生成 → 双分支特征自适应融合 → 残差连接增强 → 输出校准后的显著特征
"""

class SalientFeatureCalibratorBlock(nn.Module):
    def __init__(self, dim, r=16, L=32):
        super().__init__()
        d = max(dim // r, L)

        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        dim = x.size(1)

        # 双感受野特征提取
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)

        # 空间注意力
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)

        # 通道注意力
        ch_attn1 = self.global_pool(attn)
        z = self.fc1(ch_attn1)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, 2, dim // 2, -1)
        a_b = self.softmax(a_b)
        a1, a2 = a_b.chunk(2, dim=1)
        a1 = a1.reshape(batch_size, dim // 2, 1, 1)
        a2 = a2.reshape(batch_size, dim // 2, 1, 1)

        # 通道+空间联合加权
        w1 = a1 * agg[:, 0, :, :].unsqueeze(1)
        w2 = a2 * agg[:, 1, :, :].unsqueeze(1)
        attn = attn1 * w1 + attn2 * w2

        attn = self.conv(attn).sigmoid()
        return x * attn

if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    model = SalientFeatureCalibratorBlock(64)
    y = model(x)
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")