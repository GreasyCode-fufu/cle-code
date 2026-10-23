import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
"""
    论文地址：https://arxiv.org/pdf/2603.22794
    论文题目：It Takes Two: A Duet of Periodicity and Directionality for Burst Flicker Removal (2026 一区TOP)
    中文题目：二者共济：融合周期性与方向性的连拍图像闪烁伪影消除算法 (2026 一区TOP)
    讲解视频：https://www.bilibili.com/video/BV1TopT6oE8g/
    自相关前馈网络（Autocorrelation feed-forward network，AFFN）
        实际意义：①深层特征中的高频边缘信息丢失问题：图像中周期性干扰表现为重复的亮度/条纹变化，普通FFN只能进行一般的特征变换，难以主动发现周期规律。
                ②周期性亮暗结构容易与噪声、纹理和运动变化混在一起。
                ③普通特征增强容易把运动变化误认为亮度异常，论文实验表明，传统前馈网络在恢复时容易产生伪影。
        核心思想：利用频域自相关主动挖掘特征中的周期性重复结构，并结合频域与空间域双重增强及门控机制，强化有效周期特征、抑制无关干扰。
        代码流程：输入特征 → 1×1卷积升维 → 划分8×8局部块 → FFT频域变换与可学习频域滤波 → 功率谱计算自相关 → 频域增强 + 空间自相关增强 → 恢复原始尺寸 → 深度卷积与GELU门控融合 → 1×1卷积降维 → 输出增强特征。
"""
# 自相关前馈网络
class DFFN_AutoCorr(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DFFN_AutoCorr, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = 8
        self.dim = dim

        # 升维卷积
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 深度卷积
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        # 降维输出卷积
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # 可学习频域核
        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        # 频域增强权重
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # 空域自相关权重
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # 特征升维
        x = self.project_in(x)
        # 切成 8x8 局部块
        x_patch = rearrange(
            x, 'b c (h ph) (w pw) -> b c h w ph pw',
            ph=self.patch_size, pw=self.patch_size
        )

        # 傅里叶变换到频域
        Xf = torch.fft.rfft2(x_patch.float())
        # 频域滤波
        Xf = Xf * self.fft

        # 计算功率谱（自相关）
        power = Xf * torch.conj(Xf)
        # 逆变换得到空域自相关矩阵
        R = torch.fft.irfft2(power, s=(self.patch_size, self.patch_size))

        # 频域增强
        Xf_new = Xf + self.alpha * power
        x_patch_new = torch.fft.irfft2(Xf_new, s=(self.patch_size, self.patch_size))
        # 空域增强
        x_patch_new = x_patch_new + self.beta * R

        # 恢复原始尺寸
        x = rearrange(
            x_patch_new, 'b c h w ph pw -> b c (h ph) (w pw)',
            ph=self.patch_size, pw=self.patch_size
        )

        # 深度卷积 + 门控融合
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        # 最终输出
        x = self.project_out(x)
        return x

if __name__ == "__main__":
    x = torch.randn(1, 64, 32, 32)
    model = DFFN_AutoCorr(64, 2, False)
    y = model(x)
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")