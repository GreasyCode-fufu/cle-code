#!/usr/bin/env python3
"""EPIC-MobileNet 自检(无需数据集/权重): python selftest.py
1) 全部模块可导入, 两份注册表都含 EPIC 工厂;
2) ① 冻结语义: 可训练参数量, 冻结段 BN 在 .train() 后仍为推理态;
3) ② 凸头导出等价: sklearn 逻辑回归 -> ConvexEvidenceHead(μ/σ/Linear) 的 logits 与 sklearn 一致;
4) ③ FOV 掩码: 圆形视野覆盖率正确; 无暗背景图像上掩码池化严格退化为 GAP。
"""
import importlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def check(cond, msg):
    print(("  [OK] " if cond else "  [FAIL] ") + msg)
    if not cond:
        raise SystemExit(1)


def main():
    print("1) imports / registries")
    for mod in ("models.optimize.mbv3_cle_r6", "cle_fair_common", "evaluate", "train_cle_fair", "train", "epic_refit", "epic_gate"):
        importlib.import_module(mod)
        print(f"  [OK] import {mod}")
    try:
        importlib.import_module("models.timm_light")
        print("  [OK] import models.timm_light (timm 可用)")
    except ImportError as e:
        print(f"  [skip] models.timm_light 需要 timm: {e}")
    from cle_fair_common import MODEL_REGISTRY as R1
    from evaluate import MODEL_REGISTRY as R2
    for k in ("mobilenet_v3_large", "mbv3_r6_fz16", "mbv3_fz16_dce4m", "mbv3_dce4m"):
        check(k in R1 and k in R2, f"registry has {k} (train & evaluate)")

    from models.optimize.mbv3_cle_r6 import mbv3_r6_fz16, mbv3_dce4m, MBV3Convex, fov_weights
    print("2) ① evidence-preserving freeze")
    m = mbv3_r6_fz16(3).train()
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    check(n_tr == 3413387, f"trainable params {n_tr} == 3413387 (frozen /2–/16 = 792,488)")
    bn_eval = all(not b.training for i in range(13) for b in m.features[i].modules() if isinstance(b, nn.BatchNorm2d))
    check(bn_eval, "frozen-part BatchNorm stays in eval mode after .train()")

    print("3) ② convex head export equivalence")
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 1136)).astype(np.float32) * rng.uniform(0.1, 3, 1136).astype(np.float32) + 1.0
    y = rng.integers(0, 3, 300)
    mu, sd = X.mean(0), X.std(0)
    clf = LogisticRegression(C=0.01, max_iter=3000).fit((X - mu) / sd, y)
    net = mbv3_dce4m(3).eval()
    with torch.no_grad():
        net.classifier.mu.copy_(torch.from_numpy(mu))
        net.classifier.sigma.copy_(torch.from_numpy(sd))
        net.classifier.fc.weight.copy_(torch.from_numpy(clf.coef_).float())
        net.classifier.fc.bias.copy_(torch.from_numpy(clf.intercept_).float())
        taps = net.classifier.taps
        dims = {"f16": 960, "f12": 112, "f6": 40, "f3": 24}
        parts, o = {}, 0
        for t in taps:
            parts[t] = torch.from_numpy(X[:, o:o + dims[t]])
            o += dims[t]
        logits = net.classifier(parts).numpy()
    diff = np.abs(logits - clf.decision_function((X - mu) / sd)).max()
    check(diff < 1e-3, f"torch head logits == sklearn decision_function (max|Δ|={diff:.1e})")

    print("4) ③ FOV-aware pooling")
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    yy, xx = torch.meshgrid(torch.arange(224), torch.arange(224), indexing="ij")
    disk = (((yy - 112) ** 2 + (xx - 112) ** 2) <= 107 ** 2).float()
    img = ((0.3 * disk).expand(1, 3, 224, 224) - mean) / std
    mask = fov_weights(img)[0, 0]
    inside = torch.nn.functional.avg_pool2d(disk[None, None], 4)[0, 0] > 0.5      # 1/4 分辨率下主要落在视野内的格
    check(bool((mask[inside] == 1).all()), "FOV mask covers every in-FOV cell")
    excess = mask.mean().item() - disk.mean().item()
    check(0 <= excess <= 0.08, f"boundary dilation bounded: coverage {mask.mean().item():.3f} vs disk {disk.mean().item():.3f} (+{excess:.3f} ≤ 0.08)")
    bright = (torch.full((2, 3, 224, 224), 0.5) - mean) / std + 0.1 * torch.randn(2, 3, 224, 224)
    torch.manual_seed(0)
    a = MBV3Convex(3, taps=("f16", "f12", "f6", "f3"), pool="mgap").eval()
    b = MBV3Convex(3, taps=("f16", "f12", "f6", "f3"), pool="gap").eval()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        d = (a(bright) - b(bright)).abs().max().item()
    check(d < 1e-4, f"no dark background -> masked pooling == GAP (max|Δlogits|={d:.1e})")
    print("[selftest PASSED]")


if __name__ == "__main__":
    main()
