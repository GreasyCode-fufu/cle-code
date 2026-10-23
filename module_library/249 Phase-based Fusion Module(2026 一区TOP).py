import torch
import torch.nn as nn
"""
    论文地址：https://arxiv.org/pdf/2603.22794
    论文题目：It Takes Two: A Duet of Periodicity and Directionality for Burst Flicker Removal (2026 一区TOP)
    中文题目：二者共济：融合周期性与方向性的连拍图像闪烁伪影消除算法 (2026 一区TOP)
    讲解视频：https://www.bilibili.com/video/BV16fai6nELW/
    相位融合模块（Phase-based Fusion Module，PFM）
        实际意义：①不同帧中的异常区域位置不同：连续拍摄的多张图像中，同一位置在某一帧可能受到明暗条纹干扰，而在另一帧中可能是正常的，因此不同帧之间具有互补信息。
                ②直接融合容易引入错误信息：多个帧/图像直接进行拼接很难区分值得参考的特征。
                ③没有充分利用多帧之间的周期变化规律：论文发现周期性的明暗变化与图像的相位信息密切相关。
        核心思想：先用相位信息判断“其他帧哪些地方值得参考”，再把这些可靠信息融合到当前帧中。
        代码流程：前一帧+当前帧+后一帧特征输入 → 傅里叶转换到频域并提取相位信息 → 分别计算两路参考帧与基准帧的相位相关性 → 卷积网络生成频域权重 → 加权筛选两路参考帧频谱 → 逆傅里叶恢复空间特征 → 与基准帧通道拼接 → 3×3卷积融合 → 输出多帧融合特征。
"""
class PhaseBasedFusion(nn.Module):
    def __init__(self, channels, expansion_factor=2.66, bias=False, eps=1e-8):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels 必须为正整数，实际为 {channels}")
        hidden_channels = int(channels * expansion_factor)
        if hidden_channels <= 0:
            raise ValueError("expansion_factor 过小，导致隐藏通道数不为正")

        # 由“前一帧-基准帧”的相位相关结果生成频域权重 W0。
        self.net12 = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, 1, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=bias),
            nn.Sigmoid(),
        )
        # 由“后一帧-基准帧”的相位相关结果生成频域权重 W2。
        self.net23 = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, 1, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=bias),
            nn.Sigmoid(),
        )
        # 官方代码的融合卷积始终带偏置，因此这里不受上面的 bias 参数控制。
        self.fusion = nn.Conv2d(channels * 3, channels, 3, 1, 1)
        self.channels = channels
        self.eps = float(eps)

    def _check_inputs(self, x_prev, x_base, x_next):
        inputs = (x_prev, x_base, x_next)
        names = ("x_prev", "x_base", "x_next")
        for name, x in zip(names, inputs):
            if x.ndim != 4:
                raise ValueError(f"{name} 应为 [B, C, H, W]，实际形状为 {tuple(x.shape)}")
            if x.shape[1] != self.channels:
                raise ValueError(f"{name} 的通道数应为 {self.channels}，实际为 {x.shape[1]}")
            if not x.is_floating_point():
                raise TypeError(f"{name} 必须是浮点 Tensor，实际 dtype 为 {x.dtype}")
        if x_prev.shape != x_base.shape or x_next.shape != x_base.shape:
            raise ValueError("三路输入必须具有完全相同的 [B, C, H, W] 形状")
        if x_prev.device != x_base.device or x_next.device != x_base.device:
            raise ValueError("三路输入必须位于同一设备")
        if x_prev.dtype != x_base.dtype or x_next.dtype != x_base.dtype:
            raise ValueError("三路输入必须具有相同 dtype")

    def forward(self, x_prev, x_base, x_next):
        self._check_inputs(x_prev, x_base, x_next)
        h, w = x_base.shape[-2:]

        # rfft2 只保留实值输入频谱的非冗余部分，频谱形状为 [B, C, H, W//2+1]。
        f_prev = torch.fft.rfft2(x_prev, norm="backward")
        f_base = torch.fft.rfft2(x_base, norm="backward")
        f_next = torch.fft.rfft2(x_next, norm="backward")

        # 用 F/|F| 表示单位复数相位因子 e^(i*phi)，幅值在此被归一化掉。
        phase_prev = f_prev / (torch.abs(f_prev) + self.eps)
        phase_base = f_base / (torch.abs(f_base) + self.eps)
        phase_next = f_next / (torch.abs(f_next) + self.eps)

        # 与基准帧做复数相位相关，再取模。
        corr_prev = torch.abs(phase_prev * torch.conj(phase_base))
        corr_next = torch.abs(phase_next * torch.conj(phase_base))
        weight_prev = self.net12(corr_prev)
        weight_next = self.net23(corr_next)

        # 频域逐点加权参考帧；复频谱的相位不变、幅值由学习到的权重调制。
        f_prev_filtered = weight_prev * f_prev
        f_next_filtered = weight_next * f_next

        # 回到空间域。显式传入原尺寸，使奇数宽度也能无损恢复为 W。
        x_prev_filtered = torch.fft.irfft2(f_prev_filtered, s=(h, w), norm="backward")
        x_next_filtered = torch.fft.irfft2(f_next_filtered, s=(h, w), norm="backward")

        # 两路预筛选参考特征与未筛选基准特征拼接，再用 3x3 卷积融合回 C 通道。
        return self.fusion(torch.cat([x_prev_filtered, x_base, x_next_filtered], dim=1))

if __name__ == "__main__":
    x_prev = torch.randn(2, 64, 32, 32)
    x_base = torch.randn(2, 64, 32, 32)
    x_next = torch.randn(2, 64, 32, 32)
    model = PhaseBasedFusion(channels=64)
    y = model(x_prev, x_base, x_next)
    print("Input previous:", x_prev.shape)
    print("Input base:", x_base.shape)
    print("Input next:", x_next.shape)
    print("Output:", y.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")
