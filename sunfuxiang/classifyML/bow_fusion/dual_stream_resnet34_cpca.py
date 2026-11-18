# -*- coding: utf-8 -*-
"""
dual_stream_resnet34_cpca.py — 以 resnet34_degconv_cpca 为深度主干的专用双流融合模型

把通用双流融合模型 (dual_stream_model.DualStreamFusion) 的「语义流骨干」固定为
models/optimize/resnet34_CPCA.py 里的 resnet34_degconv_cpca
(DEGConv 浅层 layer1+2 + CPCA 深层 layer3+4 的 ResNet34 改进臂)。
classifyML 原始文件 (resnet34_CPCA.py 等) 未被修改。

为什么单独开这个文件:
  - 给「resnet34_degconv_cpca 主干」一个明确、可复制的命名入口与自检；
  - 作为 train_dual_fair.py / extract_and_classify_dual.py 的「换骨干」参考——
    你既可以直接用通用脚本传 `--model resnet34_degconv_cpca`（该模型已在
    cle_fair_common.MODEL_REGISTRY 中），也可以把本文件作为 create_model 直接 import。

骨干细节 (来自 resnet34_CPCA.py):
  - ResNet 类，self.fc = nn.Linear(512 * block.expansion, num_classes)；BasicBlock expansion=1
    → 特征维度 feat_dim = 512
  - 分类头在 `if self.include_top:` 内 (工厂默认 include_top=True)，被 DualStreamFusion
    的 _strip_classifier 识别为 `fc` 并置 Identity → forward 返回 [B, 512] 池化特征
  - 整模型参数量 ≈ 22.7M (backbone ~21.3M + 融合头 ~1.4M)

用法 A — 直接用通用脚本 (推荐，最省事):
  python bow_fusion/train_dual_fair.py --model resnet34_degconv_cpca --num_classes 3 ...

用法 B — 把本文件作为 create_model 嵌入 (需要时在 train/eval 脚本里把
  `from dual_stream_model import dual_stream as create_model` 换成
  `from dual_stream_resnet34_cpca import dual_stream_resnet34_cpca as create_model`):
  create_model(num_classes=3, bow_dim=768)   # 接口与 dual_stream 一致

自检:
  python bow_fusion/dual_stream_resnet34_cpca.py
"""

import os
import sys

# 确保 classifyML 根目录在 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from dual_stream_model import DualStreamFusion

# 固定主干名 (cle_fair_common.MODEL_REGISTRY 中的 key)
BACKBONE_NAME = "resnet34_degconv_cpca"
MODEL_NAME = "dual_resnet34_degconv_cpca"
FEAT_DIM = 512  # ResNet34 (BasicBlock expansion=1) 最终通道数


def dual_stream_resnet34_cpca(num_classes=3, bow_dim=768, d_model=256, **kwargs):
    """构造以 resnet34_degconv_cpca 为语义主干的通用双流融合模型。

    接口风格与 dual_stream(backbone_name=..., num_classes=...) 一致，只是把 backbone 锁定。
    额外接受 bow_dim / d_model。
    """
    return DualStreamFusion(backbone_name=BACKBONE_NAME, num_classes=num_classes,
                            bow_dim=bow_dim, d_model=d_model, **kwargs)


# 便于作为 create_model 直接 import (与 train_dual_fair.py 中的 create_model 同名同签名)
create_model = dual_stream_resnet34_cpca


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--num_classes', type=int, default=3)
    p.add_argument('--bow_dim', type=int, default=768)
    p.add_argument('--img_size', type=int, default=224)
    args = p.parse_args()

    m = dual_stream_resnet34_cpca(num_classes=args.num_classes, bow_dim=args.bow_dim)
    x = torch.zeros(4, 3, args.img_size, args.img_size)
    b = torch.zeros(4, args.bow_dim)
    out = m(x, b)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"[{MODEL_NAME}] backbone={BACKBONE_NAME}  feat_dim={m.feat_dim}  "
          f"head_attr={m.head_attr}  d_model={m.d_model}  bow_dim={m.bow_dim}")
    print(f"  out={tuple(out.shape)}  params={n_params/1e6:.3f}M")

    # 反向传播自检
    import torch.nn as nn
    loss = nn.CrossEntropyLoss()(out, torch.tensor([0, 1, 2, 0]))
    loss.backward()
    print(f"  backward OK (loss={loss.item():.4f})")

    # 特征提取自检 (classifier → Identity 后应输出 [B, 2*d_model])
    m.classifier = nn.Identity()
    feat = m(x, b)
    print(f"  feature-only out={tuple(feat.shape)} (expect [4, {2*m.d_model}])")
