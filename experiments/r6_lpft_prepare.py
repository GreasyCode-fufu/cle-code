#!/usr/bin/env python3
"""R6 实验 1 / LP-FT 第一阶段: 线性探针头(忠实复现官方 unlabeled_extrapolation/log_reg_sk.py)。
ImageNet 预训练 MobileNetV3-L 冻结特征(f16 GAP, 960 维; 评测同款确定性变换) ->
训练集全局标量均值/标准差归一化 -> LogisticRegression(warm_start, max_iter=200, C ∈ logspace(-7, 2, 100))
-> 按 val 准确率选 C -> 权重反归一化到原始特征空间 -> 保存为 Linear(960→K) 初始化(train_cle_fair 变换注入)。
只用 train/val。输出: experiments/r6_lpft_head_cle.pt(+ 同名 .json 审计)
"""
import copy
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
from models.MobileNetV3 import mobilenet_v3_large  # noqa: E402
from r6_p0_analysis import list_split, load_tensors, MEAN, STD, PTH  # noqa: E402

OUT = os.path.join(REPO, "experiments", "r6_lpft_head_cle.pt")


@torch.no_grad()
def f16_gap(model, X):
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
    return torch.cat([model.features((X[i:i + 128].cuda() - mean) / std).mean((2, 3)).float().cpu()
                      for i in range(0, len(X), 128)]).numpy()


def main():
    m = mobilenet_v3_large(num_classes=1000)
    m.load_state_dict(torch.load(PTH, map_location="cpu"))
    m.eval().cuda()
    (ptr, ytr), (pva, yva) = list_split("train"), list_split("val")
    Xtr, Xva = f16_gap(m, load_tensors(ptr)), f16_gap(m, load_tensors(pva))
    mu, sd = float(Xtr.mean()), float(Xtr.std())          # 官方 normalize_features: 全局标量
    Ztr, Zva = (Xtr - mu) / sd, (Xva - mu) / sd
    clf = LogisticRegression(random_state=0, warm_start=True, max_iter=200)
    best = (-1.0, None, None, None)
    for C in np.logspace(-7, 2, 100):
        clf.C = C
        clf.fit(Ztr, ytr)
        acc = float((clf.predict(Zva) == yva).mean())
        if acc > best[0]:
            best = (acc, float(C), copy.deepcopy(clf.coef_), copy.deepcopy(clf.intercept_))
    acc, C, coef, b = best
    w_raw = coef / sd                                       # logits = coef·(x−μ)/σ + b
    b_raw = b - (mu / sd) * coef.sum(axis=1)
    torch.save({"weight": torch.tensor(w_raw, dtype=torch.float32), "bias": torch.tensor(b_raw, dtype=torch.float32)}, OUT)
    chk = (Xva @ w_raw.T + b_raw).argmax(1)
    meta = {"C": C, "val_acc": round(acc, 4), "norm_scalar_mean": mu, "norm_scalar_std": sd, "feature": "ImageNet f16 GAP (960)",
            "C_grid": "logspace(-7,2,100), warm_start, max_iter=200, select by val accuracy (official log_reg_sk.py)",
            "raw_space_val_acc_check": round(float((chk == yva).mean()), 4)}
    json.dump(meta, open(OUT.replace(".pt", ".json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
