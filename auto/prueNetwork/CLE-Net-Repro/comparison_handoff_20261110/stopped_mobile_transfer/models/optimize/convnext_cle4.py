"""CLE-Net v3: SCG + PLD 组合 —— 错误分析驱动的设计闭环.

batch4 结论: OS(重叠块茎)因随机初始化损失 ImageNet 低层先验, test 0.9524 < baseline;
DSH 双尺度头待 batch4 队列出结果。已确认有效组件: SCG(条带上下文)。

v3 最终形态候选: baseline + SCG + PLD(类原型诊断亲和)。
PLD 注入点: stage3 之后、downsample_layers[3] 之前(腺体形态学层面)。
"""
import torch
import torch.nn as nn

from models.Convnext import ConvNeXt
from models.optimize.convnext_cle2 import SCGBlock
from models.optimize.pld_module import PLDBridge


class CLENet3(ConvNeXt):
    def __init__(self, scg_stages=(2, 3), use_pld=True, pld_protos=6, strip_k2=19, **kw):
        super().__init__(**kw)
        dims = [dl[1].out_channels if i > 0 else dl[0].out_channels
                for i, dl in enumerate(self.downsample_layers)]

        for si in scg_stages:
            news = []
            for blk in self.stages[si]:
                nb = SCGBlock(dim=dims[si], k2=strip_k2)
                nb.load_state_dict(blk.state_dict(), strict=False)
                news.append(nb)
            self.stages[si] = nn.Sequential(*news)

        self.use_pld = use_pld
        if use_pld:
            self.pld = PLDBridge(dim=dims[2], num_protos=pld_protos)

    def forward_features(self, x):
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)
        x = self.downsample_layers[2](x)
        x = self.stages[2](x)
        if self.use_pld:
            x = self.pld(x)                     # 诊断亲和注入(stride16)
        x = self.downsample_layers[3](x)
        x = self.stages[3](x)
        return self.norm(x.mean([-2, -1]))


def _tiny3(nc, **kw):
    return CLENet3(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=nc, **kw)


def convnext_pld(num_classes: int):
    return _tiny3(num_classes, scg_stages=(), use_pld=True)


def convnext_scg_pld(num_classes: int):
    """CLE-Net v3: baseline + B(SCG) + D(PLD)"""
    return _tiny3(num_classes, scg_stages=(2, 3), use_pld=True)
