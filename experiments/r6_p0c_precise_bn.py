#!/usr/bin/env python3
"""R6 P0c — BN 统计量滞后诊断 (2026-11-08)

假设: mbv3 BN momentum=0.01(torchvision 继承 TF decay 0.99, 为 ImageNet 长训练设计); 本协议 lr 在早停窗口内几乎
不退火(best epoch lr≈0.85–0.98×峰值), 每 epoch 仅 65 step —— running 统计量约平均最近 ~100 step, 滞后当前权重
~1.5 epoch, 选点 checkpoint 的推理态网络"权重与统计量失配", 放大 val/test 噪声。
检验: 对已训练 checkpoint 做精确 BN 重估(训练集 + 评估同款确定性变换, momentum=None 累计平均), 比较 val/test。
输出: experiments/r6_p0c_precise_bn.json
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, cohen_kappa_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
from cle_fair_common import build_model  # noqa: E402
from r6_p0_analysis import list_split, load_tensors, MEAN, STD  # noqa: E402

OUT_JSON = os.path.join(REPO, "experiments", "r6_p0c_precise_bn.json")


@torch.no_grad()
def predict(m, X):
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
    m.eval()
    return torch.cat([m((X[i:i + 128].cuda() - mean) / std).argmax(1).cpu() for i in range(0, len(X), 128)]).numpy()


@torch.no_grad()
def precise_bn(m, X):
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
    bns = [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d)]
    for b in bns:
        b.reset_running_stats()
        b.momentum = None                       # 累计平均 = 整个训练集上的精确统计量
    m.train()
    for i in range(0, len(X), 64):
        m((X[i:i + 64].cuda() - mean) / std)
    m.eval()


def main():
    splits = {sp: list_split(sp) for sp in ("train", "val", "test")}
    X = {sp: load_tensors(splits[sp][0]) for sp in splits}
    y = {sp: splits[sp][1] for sp in splits}
    res = {}
    models = sys.argv[1:] or ["mobilenet_v3_large", "mbv3_r6_star2"]
    for model in models:
        for d in sorted(glob.glob(os.path.join(REPO, "weights_cle_ac_v1_r6", f"{model}_seed*", model))):
            seed = int(d.split("_seed")[-1].split("/")[0])
            m = build_model(model, 3).cuda()
            m.load_state_dict(torch.load(os.path.join(d, "model_best.pth"), map_location="cpu"), strict=False)
            r = {}
            for tag in ("orig", "precise"):
                if tag == "precise":
                    precise_bn(m, X["train"])
                for sp in ("val", "test"):
                    p = predict(m, X[sp])
                    r[f"{sp}_{tag}"] = {"f1": round(f1_score(y[sp], p, average="macro") * 100, 2),
                                        "kappa": round(cohen_kappa_score(y[sp], p) * 100, 2),
                                        "sensM": round(float((p[y[sp] == 1] == 1).mean()) * 100, 2)}
            res.setdefault(model, {})[seed] = r
            print(f"{model:20s} s{seed} val F1 {r['val_orig']['f1']:.2f}→{r['val_precise']['f1']:.2f} | "
                  f"test F1 {r['test_orig']['f1']:.2f}→{r['test_precise']['f1']:.2f}  κ {r['test_orig']['kappa']:.2f}→"
                  f"{r['test_precise']['kappa']:.2f}  sensM {r['test_orig']['sensM']:.2f}→{r['test_precise']['sensM']:.2f}",
                  flush=True)
            del m
            torch.cuda.empty_cache()
    for model, rs in res.items():
        to = np.array([v["test_orig"]["f1"] for v in rs.values()])
        tp = np.array([v["test_precise"]["f1"] for v in rs.values()])
        print(f"[{model}] n={len(to)} test F1 orig {to.mean():.2f}±{to.std():.2f} → precise {tp.mean():.2f}±{tp.std():.2f} "
              f"(paired Δ {np.mean(tp - to):+.2f})")
    json.dump(res, open(OUT_JSON, "w"), indent=1)


if __name__ == "__main__":
    main()
