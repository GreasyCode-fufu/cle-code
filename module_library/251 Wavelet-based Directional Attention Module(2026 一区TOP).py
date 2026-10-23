import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
"""
    论文地址：https://arxiv.org/pdf/2603.22794
    论文题目：It Takes Two: A Duet of Periodicity and Directionality for Burst Flicker Removal (2026 一区TOP)
    中文题目：二者共济：融合周期性与方向性的连拍图像闪烁伪影消除算法 (2026 一区TOP)
    讲解视频：https://www.bilibili.com/video/BV1Dypx6mEwv/
    基于小波的方向注意力模块（Wavelet-based Directional Attention Module，WDAM）
        实际意义：①传统自注意力缺乏“方向性”建模：相机采用滚动快门逐行曝光，闪烁通常表现为沿水平或垂直方向分布的条纹结构。普通卷积/自注意力往往以近似各向同性的方式处理特征，难以捕获这种具有明确方向的特征。
                ②难以定位局部、细微的闪烁区域：闪烁并不一定覆盖整幅图像，有些区域只有非常轻微的亮度变化，传统注意力容易将这些区域与正常区域一起处理。
                ③自然暗区与“闪烁造成的暗区”容易混淆：闪烁不仅产生高频亮度边缘，还会形成低频暗带。如果只依赖低频亮度信息，模型很难判断暗区域究竟是场景本身较暗，还是由于闪烁导致。
        实现方式：利用小波变换将特征分解为低频与不同方向的高频分量，再利用方向性信息去引导低频特征，从而更准确地定位和去除具有水平/垂直方向特征伪影。
        代码流程：输入特征 → Haar DWT分解得到LL低频 + LH/HL/HH方向高频 → LH与HL融合生成方向引导特征 → LL生成Q/K/V并利用高频方向信息增强V → 移动窗口多头注意力增强低频特征 → 卷积优化LH/HL/HH高频细节 → IDWT融合低频与高频特征 → 输出方向增强特征。
"""
# 基于小波的方向注意力模块
class DWT_WindowAttention_SW(nn.Module):
    def __init__(self, dim, num_heads, input_resolution, window_size=8, shift_size=4, bias=False):
        super(DWT_WindowAttention_SW, self).__init__()
        # 输入特征通道数
        self.dim = dim
        # 多头注意力的头数
        self.num_heads = num_heads
        # 窗口移动步长
        self.shift_size = shift_size
        # 注意力窗口大小
        self.window_size = window_size

        # 自适应调整窗口：分辨率过小则不移动窗口
        if (input_resolution//2) <= window_size:
            self.shift_size = 0
            self.window_size = input_resolution // 2
            window_size = input_resolution // 2

        # 注意力温度系数（可学习）
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 小波分解与重构
        self.dwt = DWTForward(J=1, wave='haar')
        self.idwt = DWTInverse(wave='haar')

        # 水平+垂直高频特征处理
        self.high_conv = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1, groups=2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True)
        )

        # 三个高频分量联合优化
        self.high_out = nn.Sequential(
            nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, groups=3, bias=bias),
            nn.ReLU(inplace=True)
        )

        # 低频特征生成QKV
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        # 深度卷积提取局部特征
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        # 注意力输出投影
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1)*(2*window_size-1), num_heads)
        )

        # 预计算相对位置索引
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size)))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2*window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    # 将特征切分成不重叠窗口
    def window_partition(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        x = x.view(B, C, H//ws, ws, W//ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(-1, C, ws, ws)
        return x

    # 将窗口特征还原为原图尺寸
    def window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        C = windows.shape[1]
        ws = self.window_size
        x = windows.view(B, H//ws, W//ws, C, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, H, W)
        return x

    # 窗口循环移位
    def shift(self, x, shift_size):
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))
        return x

    # 逆循环移位
    def reverse_shift(self, x, shift_size):
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))
        return x

    # 带位置偏置的窗口注意力
    def window_attention(self, q, k, v):
        # 特征归一化
        q = F.normalize(q, dim=-2)
        k = F.normalize(k, dim=-2)

        # 计算注意力分数
        attn = torch.matmul(q.transpose(-2, -1), k)

        # 加入相对位置偏置
        N = self.window_size * self.window_size
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(N, N, -1).permute(2,0,1).unsqueeze(0)
        attn = attn + relative_position_bias

        # 温度缩放 + 归一化
        attn = attn * self.temperature
        attn = attn.softmax(dim=-1)

        # 加权值向量
        out = torch.matmul(v, attn.transpose(-2, -1))
        return out

    # 前向传播
    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 小波分解：得到低频LL 和 三个高频分量
        LL, Yh = self.dwt(x)
        Yh = Yh[0]
        LH, HL, HH = Yh[:, :, 0, :, :], Yh[:, :, 1, :, :], Yh[:, :, 2, :, :]

        # 2. 融合水平+垂直高频，生成方向权重
        filter_hv = self.high_conv(torch.cat([LH, HL], dim=1))

        # 3. 低频特征生成 QKV
        qkv = self.qkv_dwconv(self.qkv(LL))
        q, k, v_inp = qkv.chunk(3, dim=1)

        # 4. 高频权重引导 Value 向量
        v = v_inp * filter_hv + v_inp

        # 5. 窗口移位 + 切分窗口
        x_shifted = self.shift(LL, self.shift_size)
        q = self.window_partition(x_shifted)
        k = self.window_partition(x_shifted)
        v = self.window_partition(v)

        # 6. 调整维度适配多头注意力
        B_win, Cq, ws, _ = q.shape
        q = q.view(B_win, self.num_heads, Cq//self.num_heads, ws*ws)
        k = k.view(B_win, self.num_heads, Cq//self.num_heads, ws*ws)
        v = v.view(B_win, self.num_heads, Cq//self.num_heads, ws*ws)

        # 7. 窗口注意力计算
        out = self.window_attention(q, k, v)

        # 8. 窗口还原 + 移位恢复
        out = out.view(B_win, Cq, ws, ws)
        out = self.window_reverse(out, H//2, W//2)
        out = self.reverse_shift(out, self.shift_size)

        # 9. 投影输出
        out = self.project_out(out)

        # 10. 优化三个高频分量
        Yh = self.high_out(torch.cat([LH, HL, HH], dim=1))
        LH, HL, HH = Yh.chunk(3, dim=1)
        Yh = torch.stack([LH, HL, HH], dim=2)

        # 11. 小波逆变换，重建最终特征
        x_hat = self.idwt((out, [Yh]))
        return x_hat

if __name__ == "__main__":
    x = torch.randn(1, 64, 32, 32)
    model = DWT_WindowAttention_SW(64, 4, 32)
    y = model(x)
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")