# -*- coding: utf-8 -*-
"""
fasternet_cpca.py — FasterNet + CPCA 注意力的新骨干 (classifyML/bow_fusion 版)

在原始 fast_t0 (models/FasterNet.py) 的指定 stage 输出后插入 CPCA 通道-空间注意力，
**不修改 models/FasterNet.py**：本文件复用 fast_t0 的 patch_embed / stages / avgpool_pre_head / head
子模块，并在自定义 forward 里按位置插入 CPCA。

CPCA 模块采用用户提供的原始实现 (与 models/optimize/resnet34_CPCA.py 中的 CPCA 一致)：
通道注意力(avg+max) + 多尺度条带深度卷积空间注意力，所有卷积 padding='same'，空间尺寸不变、
通道数不变 → 插入后 feat_dim 仍为 1280 (与 fast_t0 一致)，可直接被双流模型去头复用。

插入位置默认 cpca_positions=(2, 3)，即 Stage2(dim=160) 与 Stage3(dim=320) 这两个深层 stage，
与 resnet34_degconv_cpca 把 CPCA 放在 layer3/layer4 的做法对应；可按需改为 (3,) / (0,1,2,3) 等。

权重兼容：fast_t0 的预训练权重 (patch_embed.* / stages.* / avgpool_pre_head.* / head.*) 可直接
载入本模型同名子模块；CPCA 为新增分支，随机初始化后随训练学习。

注册：在 bow_fusion 内通过本地扩展注册表暴露为 --model fast_t0_cpca (见 dual_stream_model.py)。

用法:
  # 直接作为双流模型的深度骨干
  python bow_fusion/train_dual_fair.py --model fast_t0_cpca --num_classes 3 ...
  # 自检
  python bow_fusion/fasternet_cpca.py
"""

import os
import sys

# 确保 classifyML 根目录在 sys.path，以便 from models.FasterNet import fast_t0
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.FasterNet import fast_t0   # 复用原始 FasterNet 构建块，不修改原文件


######################  CPCAAttention   ####     start   by  AI&CV  ###############################
class CPCAChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(CPCAChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class CPCA(nn.Module):
    def __init__(self, channels, channelAttention_reduce=4):
        super().__init__()

        self.ca = CPCAChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        inputs = self.ca(inputs)

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out
######################  CPCAAttention   ####     end   by  AI&CV  ###############################


class FasterNetCPCA(nn.Module):
    """FasterNet + CPCA：复用 fast_t0 子模块，在选定 stage 后插入 CPCA。

    Args:
        num_classes: 分类类别数
        embed_dim:   FasterNet embed_dim (fast_t0=40)
        depths:      每个 stage 的 block 数 (fast_t0=(1,2,8,2))
        cpca_positions: 在哪些 stage (0-based) 之后插入 CPCA；默认 (2,3) 即深层两个 stage
        channelAttention_reduce: CPCA 通道注意力内部缩减比
    """

    def __init__(self, num_classes=1000, embed_dim=40, depths=(1, 2, 8, 2),
                 cpca_positions=(2, 3), channelAttention_reduce=4, **kwargs):
        super().__init__()
        # 复用原始 fast_t0 的全部子模块（不修改 models/FasterNet.py）
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

        # CPCA 模块按 stage 索引命名；stage i 对应 stages-Sequential 的索引 2*i
        # (Sequential 结构: [Stage0, Merge, Stage1, Merge, Stage2, Merge, Stage3])
        self.cpca = nn.ModuleDict()
        for i in self.cpca_positions:
            dim = int(embed_dim * (2 ** i))
            self.cpca[str(i)] = CPCA(dim, channelAttention_reduce=channelAttention_reduce)
        self._seq_to_stage = {2 * i: i for i in self.cpca_positions}   # 路由用普通 dict (非子模块)

    def forward(self, x):
        x = self.patch_embed(x)
        for idx, layer in enumerate(self.stages):
            x = layer(x)
            if idx in self._seq_to_stage:          # 该位置是一个 CPCA stage 的输出
                x = self.cpca[str(self._seq_to_stage[idx])](x)
        x = self.avgpool_pre_head(x)               # B C 1 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def fast_t0_cpca(num_classes=1000, cpca_positions=(2, 3), channelAttention_reduce=4, **kwargs):
    """FasterNet + CPCA 工厂，接口风格与 fast_t0(num_classes=...) 一致。"""
    return FasterNetCPCA(num_classes=num_classes, embed_dim=40, depths=(1, 2, 8, 2),
                         cpca_positions=cpca_positions,
                         channelAttention_reduce=channelAttention_reduce, **kwargs)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--num_classes', type=int, default=3)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--cpca_positions', type=str, default='2,3',
                   help='CPCA 插入的 stage (逗号分隔)，默认 2,3')
    args = p.parse_args()
    positions = tuple(int(s) for s in args.cpca_positions.split(',') if s.strip())

    model = fast_t0_cpca(num_classes=args.num_classes, cpca_positions=positions)
    x = torch.zeros(4, 3, args.img_size, args.img_size)
    out = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    head_in = model.head.in_features if isinstance(model.head, nn.Linear) else None
    print(f"[fast_t0_cpca] cpca_positions={positions}  head_in(feat_dim)={head_in}  "
          f"out={tuple(out.shape)}  params={n_params/1e6:.3f}M")

    # 反向传播自检
    loss = nn.CrossEntropyLoss()(out, torch.tensor([0, 1, 2, 0]))
    loss.backward()
    print(f"  backward OK (loss={loss.item():.4f})")

    # 验证 CPCA 是新增分支 (state_dict 里应含 'cpca.*' 键，且 patch_embed/stages 与 fast_t0 同名)
    sd = model.state_dict()
    has_cpca = any(k.startswith("cpca.") for k in sd)
    has_stages = any(k.startswith("stages.") for k in sd)
    print(f"  state_dict: has cpca.*={has_cpca}  has stages.*={has_stages}  (共 {len(sd)} tensors)")
