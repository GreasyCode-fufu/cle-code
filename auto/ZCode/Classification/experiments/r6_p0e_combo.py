#!/usr/bin/env python3
"""R6 P0e — 证据保持 × FOV 掩码池化 × 解耦凸头 的离线组合检验 (2026-11-08)

在已训练 checkpoint(或纯 ImageNet 预训练 = "imnet")上: 冻结主干, 抽取多尺度证据(GAP 或 FOV 掩码 GAP),
强 L2 多项逻辑回归重拟合分类头(C 仅在 val 上按 Macro-F1 选), 报告 test 9 指标中的 F1/κ/BalAcc 与 sens_M。
FOV 掩码: 输入灰度 15×15 均值 > 0.03(圆形视野内组织区), 自适应平均到各尺度作为池化权重 —— 只在组织区聚合证据。
用法: python experiments/r6_p0e_combo.py imnet mobilenet_v3_large mbv3_r6_s2_fz16 ...
输出: experiments/r6_p0e_combo.json(按模型合并写入)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import glob
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, cohen_kappa_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
from cle_fair_common import build_model  # noqa: E402
from r6_p0_analysis import list_split, load_tensors, MEAN, STD, PTH  # noqa: E402
from models.MobileNetV3 import mobilenet_v3_large  # noqa: E402

OUT_JSON = os.path.join(REPO, "experiments", "r6_p0e_combo.json")
TAPS = {"f3": 3, "f6": 6, "f12": 12, "f16": 16}
CFGS = {"f16+f12|gap": (["f16", "f12"], "gap"), "f16+f12|mgap": (["f16", "f12"], "mgap"),
        "4s|gap": (["f16", "f12", "f6", "f3"], "gap"), "4s|mgap": (["f16", "f12", "f6", "f3"], "mgap")}
CGRID = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)


@torch.no_grad()
def feats(m, X):
    acts = {}
    hs = [m.features[i].register_forward_hook(lambda mm, inp, o, k=k: acts.__setitem__(k, o)) for k, i in TAPS.items()]
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
    m.eval()
    out = {f"{k}|{p}": [] for k in TAPS for p in ("gap", "mgap")}
    for i in range(0, len(X), 128):
        xb = X[i:i + 128].cuda()
        m((xb - mean) / std)
        g = xb.mean(1, keepdim=True)
        fov = (F.avg_pool2d(g, 15, 1, 7, count_include_pad=False) > 0.03).float()
        for k in TAPS:
            a = acts[k].float()
            w = F.adaptive_avg_pool2d(fov, a.shape[-2:])
            w = w / w.sum((2, 3), keepdim=True).clamp_min(1e-6)
            out[f"{k}|gap"].append(a.mean((2, 3)).cpu())
            out[f"{k}|mgap"].append((a * w).sum((2, 3)).cpu())
    for h in hs:
        h.remove()
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def refit(Xtr, ytr, Xva, yva, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best = None
    for C in CGRID:
        clf = LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr)
        f = f1_score(yva, clf.predict(Xva), average="macro")
        if best is None or f > best[0]:
            best = (f, C, clf)
    pred = best[2].predict(Xte)
    return {"C": best[1], "val_f1": round(best[0] * 100, 2), "f1": round(f1_score(yte, pred, average="macro") * 100, 2),
            "kappa": round(cohen_kappa_score(yte, pred) * 100, 2), "balacc": round(balanced_accuracy_score(yte, pred) * 100, 2),
            "sensM": round(float(recall_score(yte, pred, labels=[1], average=None)[0]) * 100, 2)}


def checkpoints(model):
    if model == "imnet":
        return [("imnet", None)]
    return [(int(d.split("_seed")[-1].split("/")[0]), d)
            for d in sorted(glob.glob(os.path.join(REPO, "weights_cle_ac_v1_r6", f"{model}_seed*", model)))]


def main():
    models = sys.argv[1:]
    splits = {sp: list_split(sp) for sp in ("train", "val", "test")}
    X = {sp: load_tensors(splits[sp][0]) for sp in splits}
    y = [splits[sp][1] for sp in ("train", "val", "test")]
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    jobs = []
    for model in models:
        res[model] = {}
        for seed, d in checkpoints(model):
            if d is None:
                m = mobilenet_v3_large(num_classes=1000)
                m.load_state_dict(torch.load(PTH, map_location="cpu"))
            else:
                m = build_model(model, 3)
                m.load_state_dict(torch.load(os.path.join(d, "model_best.pth"), map_location="cpu"), strict=False)
            m.cuda()
            Fx = {sp: feats(m, X[sp]) for sp in X}
            for c, (taps, pool) in CFGS.items():
                Xs = [np.concatenate([Fx[sp][f"{t}|{pool}"] for t in taps], 1) for sp in ("train", "val", "test")]
                jobs.append((model, str(seed), c, Xs))
            del m
            torch.cuda.empty_cache()
    outs = Parallel(n_jobs=12)(delayed(refit)(Xs[0], y[0], Xs[1], y[1], Xs[2], y[2]) for *_, Xs in jobs)
    for (model, seed, c, _), o in zip(jobs, outs):
        res[model].setdefault(seed, {})[c] = o
    for model in models:
        seeds = sorted(res[model])
        print(f"== {model} (n={len(seeds)})")
        for c in CFGS:
            v = {k: np.array([res[model][s][c][k] for s in seeds]) for k in ("f1", "kappa", "balacc", "sensM")}
            print(f"   {c:14s}: F1 {v['f1'].mean():.2f}±{v['f1'].std():.2f}  κ {v['kappa'].mean():.2f}  BalAcc {v['balacc'].mean():.2f}"
                  f"  sensM {v['sensM'].mean():.2f}  per-seed {list(v['f1'])}")
    json.dump(res, open(OUT_JSON, "w"), indent=1)


if __name__ == "__main__":
    main()
