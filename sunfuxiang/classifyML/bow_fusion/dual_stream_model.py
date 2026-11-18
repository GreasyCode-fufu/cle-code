# -*- coding: utf-8 -*-
"""
dual_stream_model.py — 通用双流注意力融合网络 (classifyML/bow_fusion 版)

结构参照 ../../BoW/models/DualStreamNet_fixed.py (语义流 + LBP-BoW 纹理流 + Cross-Attention)，
但「语义流骨干」从 BoW 版硬编码的 FasterNet 改为 cle_fair_common.MODEL_REGISTRY 中的
任意模型 (默认 fast_t0，亦可用 resnet50 / vit_base_patch16_224 / swin_* 等)。
classifyML 原始文件未被修改。

语义流: backbone(去分类头) → [B, feat_dim]  →  semantic_proj  → [B, d_model]
纹理流: BoW [B, bow_dim]                       →  texture_mlp   → [B, d_model]
融合  : Cross-Attention (Q=纹理, KV=语义) ⊕ 语义 → classifier   → [B, num_classes]

forward(x, bow_feat): 训练/推理返回 logits；
当 self.classifier 被外部置为 nn.Identity() 时返回融合特征 [B, 2*d_model]，供
extract_and_classify_dual.py 做特征提取 (feat_dim = 2*d_model，与 BoW 版的 512 一致)。
"""

import os
import sys

# 确保 classifyML 根目录在 sys.path，以便 `from cle_fair_common import build_model`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# 同时确保 bow_fusion 目录本身在 sys.path，以便 build_model 能 __import__ 兄弟模块
# (如 fasternet_cpca)，无论本文件是被「直接运行」还是「作为包导入」。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F

from cle_fair_common import build_model as _build_model_cle, MODEL_REGISTRY as _CLE_REGISTRY


# bow_fusion 本地扩展骨干 —— 不修改 cle_fair_common.py 的原始 MODEL_REGISTRY。
# value = (模块名, 工厂函数名)；模块按兄弟文件导入 (sys.path[0] = bow_fusion)。
_EXTRA_BACKBONES = {
    "fast_t0_cpca": ("fasternet_cpca", "fast_t0_cpca"),       # FasterNet + 原版 CPCA
    "fast_t0_clecpca": ("fasternet_clecpca", "fast_t0_clecpca"),  # FasterNet + CLE 改进版 CPCA
}


def build_model(name, num_classes):
    """先查 cle_fair 注册表，未命中再查 bow_fusion 本地扩展骨干。"""
    if name in _EXTRA_BACKBONES:
        module_name, func_name = _EXTRA_BACKBONES[name]
        factory = getattr(__import__(module_name, fromlist=[func_name]), func_name)
        return factory(num_classes=num_classes)
    return _build_model_cle(name, num_classes)


# 所有可选骨干 (cle_fair 注册表 ∪ bow_fusion 扩展)，供 --model 帮助文本使用
ALL_BACKBONES = sorted(set(_CLE_REGISTRY.keys()) | set(_EXTRA_BACKBONES.keys()))


def _strip_classifier(backbone):
    """移除骨干的分类头并返回特征维度 feat_dim。

    优先匹配已知分类头属性名 head / fc / classifier (与 train_cle_fair.py 加载权重时
    跳过 head/fc/classifier 的逻辑一致)；匹配到 nn.Linear → 记 in_features，置 Identity。
    否则回退「DFS 最后一个 nn.Linear」策略 (与 extract_and_classify_dual.py 一致)。
    """
    # 1) 已知属性名优先
    for attr in ("head", "fc", "classifier"):
        if hasattr(backbone, attr):
            mod = getattr(backbone, attr)
            # 找到该属性下「最后一个 nn.Linear」的 in_features 作为 feat_dim
            last_lin = None
            for m in mod.modules() if hasattr(mod, "modules") else [mod]:
                if isinstance(m, nn.Linear):
                    last_lin = m
            if last_lin is not None:
                feat_dim = last_lin.in_features
                setattr(backbone, attr, nn.Identity())
                return feat_dim, attr

    # 2) 回退: DFS 最后一个 nn.Linear
    last_name = None
    for name, mod in backbone.named_modules():
        if isinstance(mod, nn.Linear):
            last_name = name
    if last_name is None:
        raise ValueError("无法在骨干中定位分类头 (nn.Linear)，无法构建双流模型。")
    parts = last_name.split('.')
    parent = backbone
    for p in parts[:-1]:
        parent = getattr(parent, p)
    attr = parts[-1]
    feat_dim = getattr(parent, attr).in_features
    setattr(parent, attr, nn.Identity())
    return feat_dim, f"dfs:{last_name}"


class DualStreamFusion(nn.Module):
    """通用双流注意力融合网络

    Args:
        backbone_name: cle_fair_common.MODEL_REGISTRY 中的模型名 (默认 'fast_t0')
        num_classes:   分类类别数
        bow_dim:       LBP-BoW 特征维度 (默认 len(scales)*K = 3*256 = 768)
        d_model:       融合通道维度 (默认 256)
    """

    def __init__(self, backbone_name='fast_t0', num_classes=3, bow_dim=768, d_model=256):
        super().__init__()
        self.backbone_name = backbone_name
        self.d_model = d_model
        self.bow_dim = bow_dim

        # ── 语义流骨干 (注册表任意模型，去掉分类头) ──
        backbone = build_model(backbone_name, num_classes=num_classes)
        feat_dim, head_attr = _strip_classifier(backbone)
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.head_attr = head_attr

        # 语义投影: feat_dim → d_model
        self.semantic_proj = nn.Linear(feat_dim, d_model)

        # ── LBP-BoW 纹理流 ──
        self.texture_mlp = nn.Sequential(
            nn.Linear(bow_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # ── Cross-Attention 融合模块 ──
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # ── 分类头 (置 Identity 时 forward 返回融合特征) ──
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes),
        )

    def _extract_semantic(self, x):
        """骨干前向得到 [B, feat_dim] 的池化语义特征 (稳健处理非 2D 输出)。"""
        feat = self.backbone(x)
        if feat.dim() > 2:
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return feat

    def forward(self, x, bow_feat):
        """
        Args:
            x:        图像张量 [B, 3, H, W]
            bow_feat: LBP-BoW 特征 [B, bow_dim]
        Returns:
            logits [B, num_classes]；classifier 置 Identity 时返回融合特征 [B, 2*d_model]
        """
        # 语义流
        semantic_vec = self._extract_semantic(x)         # [B, feat_dim]
        semantic_tok = self.semantic_proj(semantic_vec)  # [B, d_model]
        semantic_tok = semantic_tok.unsqueeze(1)         # [B, 1, d_model]

        # 纹理流
        texture_tok = self.texture_mlp(bow_feat)         # [B, d_model]
        texture_tok = texture_tok.unsqueeze(1)           # [B, 1, d_model]

        # Cross-Attention: Q 来自纹理, K/V 来自语义
        q = self.norm_q(texture_tok)
        kv = self.norm_kv(semantic_tok)
        attn_out, _ = self.cross_attn(q, kv, kv)         # [B, 1, d_model]

        # Concat 融合 → 分类
        fused = torch.cat([attn_out, semantic_tok], dim=-1)  # [B, 1, 2*d_model]
        fused = fused.squeeze(1)                            # [B, 2*d_model]
        return self.classifier(fused)


def dual_stream(backbone_name='fast_t0', num_classes=3, bow_dim=768, d_model=256, **kwargs):
    """构造通用双流融合模型。

    接口风格对齐 train_cle_fair.py / BoW 的 create_model(num_classes=...)。
    额外接受 backbone_name / bow_dim / d_model。
    """
    return DualStreamFusion(backbone_name=backbone_name, num_classes=num_classes,
                            bow_dim=bow_dim, d_model=d_model, **kwargs)


if __name__ == '__main__':
    # 简单自检: 构造模型并跑一次前向，打印 feat_dim
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='fast_t0')
    p.add_argument('--num_classes', type=int, default=3)
    p.add_argument('--bow_dim', type=int, default=768)
    args = p.parse_args()
    m = dual_stream(backbone_name=args.model, num_classes=args.num_classes, bow_dim=args.bow_dim)
    x = torch.zeros(2, 3, 224, 224)
    b = torch.zeros(2, args.bow_dim)
    out = m(x, b)
    print(f"model={args.model} feat_dim={m.feat_dim} head_attr={m.head_attr} "
          f"out={tuple(out.shape)}")
