"""CLE-Net v4 (最终形态候选): SCG + ESA + HSC —— Sobel 显式边缘替换 PFESA.

v3 结论: PLD 原型亲和 test 倒挂(0.9253), 弃用。
v1 的 EGB(PFESA, FFT 域零参数注意力) val 高 test 反噬: 可学习门控放大了
频域噪声响应。替换为确定性 Sobel 边缘算子:
  - 3x3 Sobel 卷积核是固定的(不可学习), 提取水平/垂直梯度幅值;
  - 归一化为 [0,1] 边缘强度图, 对 dwconv 输出做乘性强调(带零门控);
  - 无 FFT、无可学习调制, 计算量更小, 且"边缘先验"与 CLE 细胞边界
    诊断语义严格对齐, 创新叙事更干净(显式结构先验 vs 黑箱注意力)。

最终形态: baseline + SCG(条带上下文) + ESA(Sobel 边缘结构注意力) + HSC(通道校准头)
= 三个创新点, 全部 CLE 数据画像驱动, 全部零门控初始等价。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Convnext import ConvNeXt, Block
from models.optimize.convnext_cle2 import SCGBlock, HSCHead, StripBranch


def _sobel_kernels(device, dtype):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device)
    return kx, ky


class ESAOp(nn.Module):
    """Sobel Edge-Structure Attention: 通道共享边缘强度图, 确定性提取。"""

    def __init__(self):
        super().__init__()

    def forward(self, x):                          # (B, C, H, W)
        B, C, H, W = x.shape
        gx = F.conv2d(x, _sobel_kernels(x.device, x.dtype)[0].expand(C, 1, 3, 3),
                      padding=1, groups=C)
        gy = F.conv2d(x, _sobel_kernels(x.device, x.dtype)[1].expand(C, 1, 3, 3),
                      padding=1, groups=C)
        mag = (gx.pow(2) + gy.pow(2) + 1e-6).sqrt()          # (B, C, H, W)
        # 通道聚合 + 全局归一 -> [0,1] 边缘强度图(逐样本)
        emap = mag.mean(dim=1, keepdim=True)                  # (B, 1, H, W)
        mn = emap.amin(dim=(2, 3), keepdim=True)
        mx = emap.amax(dim=(2, 3), keepdim=True)
        emap = (emap - mn) / (mx - mn + 1e-6)
        return emap                                              # 确定性, 无参数


class ESABlock(Block):
    """ConvNeXt Block + Sobel 边缘注意力(零门控乘性调制 dwconv 输出)。"""

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__(dim, drop_rate, layer_scale_init_value)
        self.esa = ESAOp()
        self.esa_g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x + self.esa_g * (self.esa(x) * x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class SNetBlock(Block):
    """统一叠加块: 同一块内可同时承载 ESA(乘性边缘门控) 与 SCG(零初始化条带残差)。
    修复 _swap 覆盖 bug(GATE_DIAGNOSIS.md Q01): 旧 SNet 两次整 stage 替换, 后装的
    SCG 覆盖了 stage2 的 ESA, 组合网络中 ESA 只剩 stage1。现单块按标志位叠加。"""

    def __init__(self, dim, use_esa=False, use_scg=False, drop_rate=0.,
                 layer_scale_init_value=1e-6, k2=19):
        super().__init__(dim, drop_rate, layer_scale_init_value)
        self.use_esa = use_esa
        self.use_scg = use_scg
        if use_esa:
            self.esa = ESAOp()
            self.esa_g = nn.Parameter(torch.zeros(1))           # 标量门: 高 lr 组训练
        if use_scg:
            self.scg = StripBranch(dim, k2)                     # c3 零初始化 => 恒等

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        if self.use_esa:
            x = x + self.esa_g * (self.esa(x) * x)
        if self.use_scg:
            x = x + self.scg(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class SNet(ConvNeXt):
    """ConvNeXt + SCG + ESA + HSC 渐进叠加。"""

    def __init__(self, esa_stages=(1, 2), scg_stages=(2, 3), use_hsc=True, **kw):
        super().__init__(**kw)
        dims = [dl[1].out_channels if i > 0 else dl[0].out_channels
                for i, dl in enumerate(self.downsample_layers)]

        for si in sorted(set(esa_stages) | set(scg_stages)):
            news = []
            for blk in self.stages[si]:
                nb = SNetBlock(dim=dims[si],
                               use_esa=(si in esa_stages),
                               use_scg=(si in scg_stages))
                nb.load_state_dict(blk.state_dict(), strict=False)
                news.append(nb)
            self.stages[si] = nn.Sequential(*news)

        self.use_hsc = use_hsc
        if use_hsc:
            self.hsc = HSCHead(dims[-1])

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x.mean([-2, -1]))
        if self.use_hsc:
            x = self.hsc(x)
        return x


def _tinys(nc, **kw):
    return SNet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=nc, **kw)


def convnext_esa(num_classes: int):
    return _tinys(num_classes, esa_stages=(1, 2), scg_stages=(), use_hsc=False)


def convnext_esa_scg_hsc(num_classes: int):
    """S-Net 最终形态: baseline + ESA + SCG + HSC"""
    return _tinys(num_classes, esa_stages=(1, 2), scg_stages=(2, 3), use_hsc=True)
