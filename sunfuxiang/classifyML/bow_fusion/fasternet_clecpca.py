# -*- coding: utf-8 -*-
"""
fasternet_clecpca.py — FasterNet + CLECPCA 注意力的新骨干 (classifyML/bow_fusion 版)

【本次修订要点】
  1. CLECPCA 改为「残差 + proj_out 零初始化」：模块在初始时刻 ≈ 恒等，载入 fast_t0
     预训练权重后不会被随机分支破坏，收敛更稳（思路同 ReZero / zero-init gating）。
  2. 条带卷积核按 stage 深度自适应：越深的 stage 特征图越小 (14x14 / 7x7)，大核会大量
     卷到 padding 的 0；现按 stage 给不同 band_kernels，避免「多尺度」在深层塌缩。
  3. 新增显式 forward_features()，forward() 复用之；与 dual_stream 的 head→Identity
     机制并存（两种取特征方式都成立）。

与 fasternet_cpca.py 平行：同样不修改 models/FasterNet.py，复用 fast_t0 的
patch_embed / stages / avgpool_pre_head / head 子模块。CLECPCA 不改变空间尺寸与通道数
→ feat_dim 仍为 1280。

注册：在 bow_fusion 本地扩展注册表中暴露为 --model fast_t0_clecpca (见 dual_stream_model.py)。

用法:
  python bow_fusion/train_dual_fair.py --model fast_t0_clecpca --num_classes 3 ...
  python bow_fusion/fasternet_clecpca.py                 # 自检
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.FasterNet import fast_t0   # 复用原始 FasterNet 构建块，不修改原文件


# ===================== CLECPCA（针对 CLE 改进的 CPCA）=====================
class CLEChannelAttention(nn.Module):
    """通道注意力：在 avg / max 基础上增加 std(标准差) 池化，对纹理更敏感。"""

    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        internal_neurons = max(internal_neurons, 8)  # 防止瓶颈过小退化
        self.fc1 = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def _mlp(self, pooled):
        return self.fc2(self.act(self.fc1(pooled)))

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, output_size=1)          # E[x]   全局均值
        mx = F.adaptive_max_pool2d(x, output_size=1)           # 最显著响应
        var = F.adaptive_avg_pool2d(x * x, output_size=1) - avg * avg
        std = torch.sqrt(var.clamp_min(1e-5))                  # 纹理能量/对比度

        w = (torch.sigmoid(self._mlp(avg))
             + torch.sigmoid(self._mlp(mx))
             + torch.sigmoid(self._mlp(std)))
        return x * w


class CLECPCA(nn.Module):
    """
    针对 CLE 灰度图改进的 CPCA（残差版）。

    Args:
        channels:     输入/输出通道数
        reduce:       通道注意力瓶颈的压缩比
        band_kernels: 条带卷积尺度元组，如 (7, 11)；每个 k 生成 (1,k)+(k,1) 两步深度卷积。
                      建议随 stage 深度（特征图变小）取更小的核。
    """

    def __init__(self, channels, reduce=4, band_kernels=(7, 11)):
        super().__init__()
        self.band_kernels = tuple(band_kernels)

        self.ca = CLEChannelAttention(channels, channels // reduce)

        # 全局感知器
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

        # 基础局部特征 (5x5) 与细节分支 (3x3)
        self.dwconv5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dwconv3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

        # 多尺度条带卷积（按 band_kernels 动态构建）
        self.band_h = nn.ModuleList()
        self.band_v = nn.ModuleList()
        for k in self.band_kernels:
            self.band_h.append(
                nn.Conv2d(channels, channels, (1, k), padding=(0, k // 2), groups=channels))
            self.band_v.append(
                nn.Conv2d(channels, channels, (k, 1), padding=(k // 2, 0), groups=channels))

        # 可学习的分支融合权重：[x_init, detail, *bands]
        n_branch = 2 + len(self.band_kernels)
        self.branch_w = nn.Parameter(torch.ones(n_branch))

        # 独立的空间权重卷积与输出卷积
        self.spatial = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

        # proj_out 零初始化 → 残差初始 ≈ 恒等，保护预训练特征
        nn.init.zeros_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        identity = x

        # 全局感知 + 通道优先
        h = self.act(self.proj_in(x))
        h = self.ca(h)

        # 多尺度空间分支
        x_init = self.dwconv5(h)
        detail = self.dwconv3(h) - x_init                      # 近似带通/高频，保细胞级纹理
        branches = [x_init, detail]
        for conv_h, conv_v in zip(self.band_h, self.band_v):
            branches.append(conv_v(conv_h(x_init)))

        # 自适应融合
        w = torch.softmax(self.branch_w, dim=0)
        fused = sum(wi * bi for wi, bi in zip(w, branches))

        # 生成空间权重并调制；残差回接，初始时刻不破坏输入
        att = self.spatial(fused)
        out = identity + self.proj_out(att * h)
        return out


# ===================== FasterNet + CLECPCA 包装 =====================
# 各 stage 默认条带核：特征图越深越小（按 224 输入估算的尺寸）
#   stage0 ~56x56, stage1 ~28x28, stage2 ~14x14, stage3 ~7x7
_DEFAULT_BAND_KERNELS = {0: (11, 21), 1: (7, 11), 2: (5, 7), 3: (3, 5)}


class FasterNetCLECPCA(nn.Module):
    """FasterNet + CLECPCA：复用 fast_t0 子模块，在选定 stage 后插入 CLECPCA。

    Args:
        num_classes:    分类类别数
        embed_dim:      FasterNet embed_dim (fast_t0=40)
        depths:         每个 stage 的 block 数 (fast_t0=(1,2,8,2))
        cpca_positions: 在哪些 stage (0-based) 之后插入 CLECPCA；默认 (2,3)
        reduce:         CLECPCA 通道注意力压缩比
        band_kernels:   None=按 stage 用 _DEFAULT_BAND_KERNELS；传元组则所有位置统一使用
    """

    def __init__(self, num_classes=1000, embed_dim=40, depths=(1, 2, 8, 2),
                 cpca_positions=(2, 3), reduce=4, band_kernels=None, **kwargs):
        super().__init__()
        backbone = fast_t0(num_classes=num_classes)
        self.patch_embed = backbone.patch_embed
        self.stages = backbone.stages
        self.avgpool_pre_head = backbone.avgpool_pre_head
        self.head = backbone.head

        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.cpca_positions = tuple(cpca_positions)
        for i in self.cpca_positions:
            if not (0 <= i < self.num_stages):
                raise ValueError(f"cpca_positions stage {i} 越界，合法范围 0~{self.num_stages - 1}")

        # CLECPCA 模块按 stage 索引命名；stage i 对应 stages-Sequential 的索引 2*i
        self.clecpca = nn.ModuleDict()
        for i in self.cpca_positions:
            dim = int(embed_dim * (2 ** i))
            ks = band_kernels if band_kernels is not None else _DEFAULT_BAND_KERNELS.get(i, (5, 7))
            self.clecpca[str(i)] = CLECPCA(dim, reduce=reduce, band_kernels=ks)
        self._seq_to_stage = {2 * i: i for i in self.cpca_positions}

    def forward_features(self, x):
        """返回过 head 之前的 1280 维特征（供双流/特征提取直接调用）。"""
        x = self.patch_embed(x)
        for idx, layer in enumerate(self.stages):
            x = layer(x)
            if idx in self._seq_to_stage:
                x = self.clecpca[str(self._seq_to_stage[idx])](x)
        x = self.avgpool_pre_head(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)


def fast_t0_clecpca(num_classes=1000, cpca_positions=(2, 3), reduce=4, band_kernels=None, **kwargs):
    """FasterNet + CLECPCA 工厂，接口风格与 fast_t0(num_classes=...) 一致。

    注意：本工厂不自动加载预训练权重；如需载入 fast_t0 预训练，请在外部用
          load_state_dict(ckpt, strict=False) 处理（clecpca.* 为新增项会被跳过）。
    """
    return FasterNetCLECPCA(num_classes=num_classes, embed_dim=40, depths=(1, 2, 8, 2),
                            cpca_positions=cpca_positions, reduce=reduce,
                            band_kernels=band_kernels, **kwargs)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--num_classes', type=int, default=3)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--cpca_positions', type=str, default='2,3')
    p.add_argument('--band_kernels', type=str, default='',
                   help="留空=按 stage 自适应；或 '5,7' 强制统一")
    args = p.parse_args()
    positions = tuple(int(s) for s in args.cpca_positions.split(',') if s.strip())
    bk = tuple(int(s) for s in args.band_kernels.split(',') if s.strip()) or None

    model = fast_t0_clecpca(num_classes=args.num_classes, cpca_positions=positions,
                            band_kernels=bk)
    x = torch.zeros(4, 3, args.img_size, args.img_size)

    # 残差零初始化自检：未训练时 forward_features 应等价于无注意力骨干（这里仅验证可前向）
    out = model(x)
    feat = model.forward_features(x)
    n_params = sum(pp.numel() for pp in model.parameters())
    head_in = model.head.in_features if isinstance(model.head, nn.Linear) else None
    print(f"[fast_t0_clecpca] positions={positions} band_kernels={bk or '自适应'} "
          f"feat_dim={tuple(feat.shape)} head_in={head_in} out={tuple(out.shape)} "
          f"params={n_params/1e6:.3f}M")

    loss = nn.CrossEntropyLoss()(out, torch.tensor([0, 1, 2, 0]))
    loss.backward()
    print(f"  backward OK (loss={loss.item():.4f})")

    sd = model.state_dict()
    has_cle = any(k.startswith("clecpca.") for k in sd)
    has_stages = any(k.startswith("stages.") for k in sd)
    print(f"  state_dict: has clecpca.*={has_cle}  has stages.*={has_stages}  (共 {len(sd)} tensors)")