#!/usr/bin/env python3
"""R6 P0d — 解耦凸头重拟合(decoupled convex head re-fit)离线检验 (2026-11-08)

问题(P0-F1): 端到端训练的冷启动头决策面差。检验: 对已训练的 checkpoint(选点不变), 冻结主干, 在多尺度 GAP 证据上
用强 L2 多项逻辑回归重拟合分类头(C 在 val 上按 Macro-F1 选, 仅用 train/val, 不触碰 test), 与网络自身 test 对比。
参照: Kang et al. ICLR 2020(decoupling: 表征与分类器解耦, cRT); Kumar et al. ICLR 2022(LP-FT)。
输出: experiments/r6_p0d_refit.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import glob
import json
import sys

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
from cle_fair_common import build_model  # noqa: E402
from r6_p0_analysis import list_split, load_tensors, MEAN, STD  # noqa: E402

OUT_JSON = os.path.join(REPO, "experiments", "r6_p0d_refit.json")
TAPS = {"f3": 3, "f6": 6, "f12": 12, "f16": 16}
CFGS = {"f16": ["f16"], "f16+f12": ["f16", "f12"], "f16+f12+f6+f3": ["f16", "f12", "f6", "f3"]}
CGRID = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)


@torch.no_grad()
def gap_feats(m, X):
    acts = {}
    hs = [m.features[i].register_forward_hook(lambda mm, inp, o, k=k: acts.__setitem__(k, o)) for k, i in TAPS.items()]
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
    m.eval()
    out = {k: [] for k in TAPS}
    for i in range(0, len(X), 128):
        m((X[i:i + 128].cuda() - mean) / std)
        for k in TAPS:
            out[k].append(acts[k].float().mean((2, 3)).cpu())
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
    p = best[2].predict_proba(Xte)
    pred = p.argmax(1)
    bm = yte != 2
    return {"C": best[1], "val_f1": round(best[0] * 100, 2), "test_f1": round(f1_score(yte, pred, average="macro") * 100, 2),
            "kappa": round(cohen_kappa_score(yte, pred) * 100, 2),
            "sensM": round(float(recall_score(yte, pred, labels=[1], average=None)[0]) * 100, 2),
            "auc_bm": round(float(roc_auc_score(yte[bm] == 1, p[bm, 1] / (p[bm, 0] + p[bm, 1] + 1e-12))), 4)}


def net_test(d):
    for f in glob.glob(os.path.join(d, "eval_*", "results.json")):
        r = json.load(open(f))
        if "test" in r["test_dir"]:
            return {"test_f1": round(r["macro_f1"] * 100, 2), "kappa": round(r["kappa"] * 100, 2),
                    "sensM": round(r["per_class"]["Malignant_lesions"]["sensitivity"] * 100, 2)}


def main():
    models = sys.argv[1:] or ["mobilenet_v3_large", "mbv3_r6_star2"]
    splits = {sp: list_split(sp) for sp in ("train", "val", "test")}
    X = {sp: load_tensors(splits[sp][0]) for sp in splits}
    y = [splits[sp][1] for sp in ("train", "val", "test")]
    res, jobs = {}, []
    for model in models:
        for d in sorted(glob.glob(os.path.join(REPO, "weights_cle_ac_v1_r6", f"{model}_seed*", model))):
            seed = int(d.split("_seed")[-1].split("/")[0])
            m = build_model(model, 3).cuda()
            m.load_state_dict(torch.load(os.path.join(d, "model_best.pth"), map_location="cpu"), strict=False)
            F_ = {sp: gap_feats(m, X[sp]) for sp in X}
            res.setdefault(model, {})[seed] = {"net": net_test(d)}
            for c, parts in CFGS.items():
                Xs = [np.concatenate([F_[sp][k] for k in parts], 1) for sp in ("train", "val", "test")]
                jobs.append((model, seed, c, Xs))
            del m
            torch.cuda.empty_cache()
    outs = Parallel(n_jobs=12)(delayed(refit)(Xs[0], y[0], Xs[1], y[1], Xs[2], y[2]) for *_, Xs in jobs)
    for (model, seed, c, _), o in zip(jobs, outs):
        res[model][seed][c] = o
    for model, rs in res.items():
        seeds = sorted(rs)
        print(f"== {model} (n={len(seeds)})")
        net = np.array([rs[s]["net"]["test_f1"] for s in seeds])
        print(f"   network head : F1 {net.mean():.2f}±{net.std():.2f}  κ {np.mean([rs[s]['net']['kappa'] for s in seeds]):.2f}"
              f"  sensM {np.mean([rs[s]['net']['sensM'] for s in seeds]):.2f}   per-seed {list(net)}")
        for c in CFGS:
            v = np.array([rs[s][c]["test_f1"] for s in seeds])
            print(f"   refit {c:14s}: F1 {v.mean():.2f}±{v.std():.2f} (paired Δ {np.mean(v - net):+.2f}, {int(np.sum(v > net))}/{len(v)} seeds ↑)"
                  f"  κ {np.mean([rs[s][c]['kappa'] for s in seeds]):.2f}  sensM {np.mean([rs[s][c]['sensM'] for s in seeds]):.2f}"
                  f"  C={[rs[s][c]['C'] for s in seeds]}  per-seed {list(v)}")
    json.dump(res, open(OUT_JSON, "w"), indent=1)


if __name__ == "__main__":
    main()
