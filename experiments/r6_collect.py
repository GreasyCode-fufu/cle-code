#!/usr/bin/env python3
"""R6 结果汇总: 读 weights 根下各模型各 seed 的 test eval results.json(evaluate.py 输出),
给出 9 指标 + 逐类敏感度的 均值±总体标准差(n 注明) 与相对参照模型的 同 seed 配对 Δ。
用法: python experiments/r6_collect.py [--root weights_cle_ac_v1_r6] [--ref mobilenet_v3_large] [model ...]
"""
import argparse
import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = [("macro_f1", "F1"), ("accuracy", "Acc"), ("weighted_f1", "wF1"), ("balanced_accuracy", "BalAcc"),
           ("macro_sensitivity", "mSens"), ("macro_specificity", "mSpec"), ("kappa", "Kappa"), ("mcc", "MCC"),
           ("auc_roc_macro", "AUROC")]


def load_runs(root, model):
    runs = {}
    for d in sorted(glob.glob(os.path.join(REPO, root, f"{model}_seed*", model))):
        seed = int(d.split("_seed")[-1].split("/")[0])
        for f in sorted(glob.glob(os.path.join(d, "eval_*", "results.json"))):
            r = json.load(open(f))
            if "test" in r.get("test_dir", "") and "val" not in os.path.basename(r.get("test_dir", "")):
                m = {k: r[k] for k, _ in METRICS}
                per = r.get("per_class", {})
                cls = list(per.keys())
                for c in cls:
                    m[f"sens_{c[:1]}"] = per[c].get("sensitivity", np.nan)
                runs[seed] = m
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--root", default="weights_cle_ac_v1_r6")
    ap.add_argument("--ref", default="mobilenet_v3_large")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if not a.models:
        a.models = sorted({os.path.basename(p).rsplit("_seed", 1)[0]
                           for p in glob.glob(os.path.join(REPO, a.root, "*_seed*"))})
    ref = load_runs(a.root, a.ref)
    keys = [k for k, _ in METRICS] + ["sens_B", "sens_M", "sens_b"]
    names = [n for _, n in METRICS] + ["sensB", "sensM", "sensN"]
    print(f"{'model':22s} n  " + " ".join(f"{n:>13s}" for n in names))
    out = {}
    for model in [a.ref] + [m for m in a.models if m != a.ref]:
        runs = load_runs(a.root, model)
        if not runs:
            continue
        seeds = sorted(runs)
        row, rec = [], {"seeds": seeds}
        for k in keys:
            v = np.array([runs[s].get(k, np.nan) for s in seeds]) * 100
            paired = [s for s in seeds if s in ref]
            d = np.array([runs[s][k] - ref[s][k] for s in paired]) * 100 if model != a.ref and paired else None
            rec[k] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                      "delta": float(d.mean()) if d is not None else None,
                      "per_seed": {s: round(runs[s][k] * 100, 2) for s in seeds}}
            row.append(f"{np.nanmean(v):6.2f}" + (f"({d.mean():+5.2f})" if d is not None else "       "))
        out[model] = rec
        print(f"{model:22s} {len(seeds)}  " + " ".join(f"{c:>13s}" for c in row))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
