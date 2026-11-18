"""R6 实验 1: 最接近 EPIC 的微调策略对比方法 —— 在统一协议(AdamW 1e-4 / wd 5e-2 / cosine / bs32 / 增广 /
val Macro-F1 选点 / 早停 30 / seeds 42–46)内忠实复现。官方代码见 comparison/third_party/(只读参考)。

- LP-FT(Kumar et al., ICLR 2022; AnanyaKumar/transfer_learning):
    先在 ImageNet 冻结特征(f16 GAP, 960 维)上做线性探针: 特征按训练集全局标量均值/标准差归一化,
    LogisticRegression(warm_start, C ∈ logspace(-7, 2, 100)), 按 val 准确率选 C(官方 log_reg_sk.py);
    权重反归一化后作为分类头初始化(官方 set_last_layer), 再全网络微调。头部 = Dropout0.2 + Linear(960→K),
    与端到端线性头 mbv3_r6_lin 同构 —— 两者只差头部初始化。LP 头由 experiments/r6_lpft_prepare.py 生成,
    经 train_cle_fair 的 checkpoint 变换注入。
- L2-SP(Li, Grandvalet, Davoine, ICML 2018; holyseven/TransferLearningClassification):
    Ω = α·½Σ‖w − w⁰‖²(预训练层权重, 官方仅对 'weights' 变量即 conv/fc 核)+ β·½Σ‖w‖²(新层权重)。
    官方默认 α=β=0.01; 论文亦用 α=0.1 —— 两档各 5 seeds, 按 val 选。起点 w⁰ 由 checkpoint 变换注入冻结副本 sp_ref。
    注: 协议的 AdamW 解耦 wd(5e-2, lr 1e-4 下 100 epoch 约收缩 3%)保持不变, L2-SP 项作为损失加入。
- Surgical fine-tuning(Lee et al., ICLR 2023; anniesch/surgical-finetuning):
    只训练一个块, 其余参数冻结; 官方整体 model.train() —— 冻结块的 BN 统计量仍随目标域更新(不同于 EPIC ①)。
    官方任务与源任务同标签空间, 头部可不训; 本任务为新 3 类, 头部为随机初始化, 故各变体均训练头部。
    块划分(对应官方 ResNet conv1+layer1 / layer2 / layer3 / layer4 / fc):
      b1 = features[0..3](/2–/4), b2 = [4..6](/8), b3 = [7..12](/16), b4 = [13..16](/32), head = 仅头部。
- cRT / τ-norm(Kang et al., ICLR 2020; facebookresearch/classifier-balancing): 后处理, 见 experiments/r6_crt.py。
"""
import copy
from types import SimpleNamespace

import torch.nn as nn

from models.optimize.mbv3_cle_r6 import MBV3R6
from models.MobileNetV3 import mobilenet_v3_large


class MBV3L2SP(MBV3R6):
    """原 MLP 头 + L2-SP 正则(经 train_cle_fair 既有辅助损失钩子 loss += mfm_lambda * mfm.loss 接入)。"""

    def __init__(self, num_classes, alpha, beta=0.01):
        super().__init__(num_classes, head="mlp")
        self.l2sp_alpha, self.l2sp_beta = alpha, beta
        self.sp_ref = copy.deepcopy(self.features)        # 预训练起点 w⁰(checkpoint 变换注入, 冻结, 不前向)
        self.sp_ref.requires_grad_(False)
        self.mfm_lambda = 1.0
        self._aux = None

    @property
    def mfm(self):
        return SimpleNamespace(loss=self._aux)

    def forward(self, x):
        out = super().forward(x)
        if self.training:
            l_sp = sum(((p - r) ** 2).sum() for p, r in zip(self.features.parameters(), self.sp_ref.parameters())
                       if p.dim() > 1)
            l_new = sum((p ** 2).sum() for p in self.classifier.parameters() if p.dim() > 1)
            self._aux = 0.5 * self.l2sp_alpha * l_sp + 0.5 * self.l2sp_beta * l_new
        return out


class MBV3Surgical(MBV3R6):
    """原 MLP 头; 只训练所选块 + 头部, 其余参数冻结; BN 训练态(统计量更新, 同官方)。"""
    BLOCKS = {"b1": range(0, 4), "b2": range(4, 7), "b3": range(7, 13), "b4": range(13, 17), "head": range(0)}

    def __init__(self, num_classes, block):
        super().__init__(num_classes, head="mlp")
        keep = set(self.BLOCKS[block])
        for i, blk in enumerate(self.features):
            if i not in keep:
                blk.requires_grad_(False)


def mbv3_lpft(num_classes=3):
    return MBV3R6(num_classes, head="lin")


def mbv3_l2sp_a01(num_classes=3):
    return MBV3L2SP(num_classes, alpha=0.1)


def mbv3_l2sp_a001(num_classes=3):
    return MBV3L2SP(num_classes, alpha=0.01)


def mbv3_sft_b1(num_classes=3):
    return MBV3Surgical(num_classes, "b1")


def mbv3_sft_b2(num_classes=3):
    return MBV3Surgical(num_classes, "b2")


def mbv3_sft_b3(num_classes=3):
    return MBV3Surgical(num_classes, "b3")


def mbv3_sft_b4(num_classes=3):
    return MBV3Surgical(num_classes, "b4")


def mbv3_sft_head(num_classes=3):
    return MBV3Surgical(num_classes, "head")


def mbv3_crt(num_classes=3):
    """cRT 输出结构: 冻结的 baseline 主干 + 重新训练的线性头(Dropout 在推理时为恒等)。"""
    return MBV3R6(num_classes, head="lin")


def mbv3_taunorm(num_classes=3):
    """τ-norm 输出结构: 与 baseline 相同(仅末层权重按类范数^τ 缩放, 偏置置 0)。"""
    return mobilenet_v3_large(num_classes=num_classes)
