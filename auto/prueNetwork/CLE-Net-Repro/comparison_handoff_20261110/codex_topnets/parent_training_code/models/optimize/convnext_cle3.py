"""CLE-Net v2: 结构级创新 —— 2026-10-25 v3 设计 (test 反馈驱动)

batch1-3 结论: 插件式创新在 val 上普遍 0.98+ 但 test 收益有限(EGB 反噬, HSC 不稳),
SCG(条带上下文)是唯一稳定正收益。val/test 分布差距是主要矛盾。
v2 转向「改造 baseline 自身结构」, 三个创新点均由 CLE 数据画像直接驱动:

  A. OS  Overlapping Stem —— 重叠块茎。
     原 ConvNeXt 的 4x4/stride4 不重叠 patchify 一次性丢弃块内高频细节,
     而 CLE 图像 85%+ 能量在高频(细胞边界/隐窝纹理正是诊断依据)。
     改为 3x3/s2 + 3x3/s2 重叠卷积块茎, 逐级提取, 保留细胞级细节。
  B. SCG Strip Context Guidance —— (承袭 v2 已验证, +0.35pp)
     腺体隐窝的各向异性长程结构先验。
  C. DSH Dual-Scale Head —— 双尺度决策头。
     ConvNeXt 仅用 stage4(stride32) 全局特征分类; CLE 诊断同时依赖
     细胞级细节(stage3, stride16)与腺体结构级(stage4)线索,
     两尺度 GAP 特征各自归一化后拼接进分类头。

工程: 原 downsample_layers[0] 保留在模块树中但不再参与前向(ostem 接管),
其权重正常加载(惰性参数, 4.6k, 可忽略), 零 transform 零形状冲突。
新增段: ostem / norm_s3 (head 属分类头, 白名单自动覆盖)。
"""
import torch
import torch.nn as nn

from models.Convnext import ConvNeXt, LayerNorm
from models.optimize.convnext_cle2 import SCGBlock


class OverlapStem(nn.Module):
    """3x3/s2 + GELU + 3x3/s2 重叠卷积块茎, 输出尺寸与原版 4x4/s4 一致。"""

    def __init__(self, dim=96):
        super().__init__()
        mid = dim // 2
        self.c1 = nn.Conv2d(3, mid, 3, stride=2, padding=1)
        self.n1 = LayerNorm(mid, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()
        self.c2 = nn.Conv2d(mid, dim, 3, stride=2, padding=1)
        self.n2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        return self.n2(self.c2(self.act(self.n1(self.c1(x)))))


class CLENet2(ConvNeXt):
    """ConvNeXt + {A:OS, B:SCG@stages, C:DSH} 渐进叠加。"""

    def __init__(self, use_os=False, scg_stages=(), use_dsh=False,
                 strip_k2=19, **kw):
        super().__init__(**kw)
        dims = [dl[1].out_channels if i > 0 else dl[0].out_channels
                for i, dl in enumerate(self.downsample_layers)]

        self.use_os = use_os
        if use_os:
            self.ostem = OverlapStem(dims[0])

        if scg_stages:
            for si in scg_stages:
                news = []
                for blk in self.stages[si]:
                    nb = SCGBlock(dim=dims[si], k2=strip_k2)
                    nb.load_state_dict(blk.state_dict(), strict=False)
                    news.append(nb)
                self.stages[si] = nn.Sequential(*news)

        self.use_dsh = use_dsh
        if use_dsh:
            self.norm_s3 = nn.LayerNorm(dims[2], eps=1e-6)
            self.head = nn.Linear(dims[2] + dims[3], self.head.out_features)

    def forward_features(self, x):
        x = self.ostem(x) if self.use_os else self.downsample_layers[0](x)
        x = self.stages[0](x)
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)
        x = self.downsample_layers[2](x)
        x = self.stages[2](x)                       # stride16, 细胞级
        gap3 = x.mean([-2, -1]) if self.use_dsh else None
        x = self.downsample_layers[3](x)
        x = self.stages[3](x)                       # stride32, 结构级
        gap4 = self.norm(x.mean([-2, -1]))
        if self.use_dsh:
            return torch.cat([self.norm_s3(gap3), gap4], dim=-1)
        return gap4


def _tiny2(nc, **kw):
    return CLENet2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                   num_classes=nc, **kw)


def convnext_os(num_classes: int):
    return _tiny2(num_classes, use_os=True)


def convnext_dsh(num_classes: int):
    return _tiny2(num_classes, use_dsh=True)


def convnext_os_scg(num_classes: int):
    return _tiny2(num_classes, use_os=True, scg_stages=(2, 3))


def convnext_os_scg_dsh(num_classes: int):
    """CLE-Net v2 最终形态: baseline + A(OS) + B(SCG) + C(DSH)"""
    return _tiny2(num_classes, use_os=True, scg_stages=(2, 3), use_dsh=True)
