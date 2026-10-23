import torch
import torch.nn as nn
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

"""
    论文地址：https://arxiv.org/pdf/2603.28584v1
    论文题目：ORSIFlow: Saliency-Guided Rectified Flow for Optical Remote Sensing Salient Object Detection (ICME 2026)
    中文题目：ORSIFlow：面向光学遥感显著目标检测的显著性引导矫正流方法（ICME 2026）
    讲解视频：https://www.bilibili.com/video/BV1m2nF6YEER/
    显著特征判别器（Salient Feature Discriminator，SFD）
        实际意义：①复杂背景干扰导致目标与背景语义混淆问题：光学遥感图像通常具有大范围复杂背景；地物纹理高度相似；目标与背景颜色、结构接近。传统方法直接预测显著图时，容易受到背景区域影响，导致错误检测，背景误激活，目标区域不完整。②缺少全局语义关系建模能力的问题：遥感具有较大空间分布范围，仅依靠局部卷积特征难以理解目标整体结构。例如：一条河流具有长距离连续关系；道路具有明显空间拓扑结构；大面积建筑群存在区域关联。因此模型需要捕获不同特征通道之间的全局语义依赖关系。
        核心思想：通过动态图关系建模挖掘通道间全局语义依赖，增强显著目标特征表达并抑制复杂背景干扰。
        代码流程：输入特征 → 1×1卷积生成多分支通道描述 → GAP提取全局通道语义 → 通道差异计算生成关系矩阵 → 构建动态通道图并进行图卷积推理 → 输出增强后的显著特征
"""

# 显著特征判别器
class SalientFeatureDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(SalientFeatureDiscriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 自动设置通道数
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction

        # 1x1 卷积层
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x, A=None, alpha=1):
        # 三个卷积分支
        x1 = self.conv1(x).mean(-2)
        x2 = self.conv2(x).mean(-2)
        x3 = self.conv3(x)

        # 计算特征相关性
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha

        # 加入外部注意力矩阵
        if A is not None:
            x1 = x1 + A.unsqueeze(0).unsqueeze(0)

        # 爱因斯坦求和做特征融合
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

if __name__ == "__main__":
    A = torch.randn(50, 50)
    x = torch.randn(1, 32, 50, 50)
    model = SalientFeatureDiscriminator(32, 32)
    y = model(x,A)
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")