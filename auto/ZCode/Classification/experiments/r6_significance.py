#!/usr/bin/env python3
"""R6 主域显著性与误差结构: 最终网络 vs baseline(同 seed 配对, 逐图预测来自 evaluate.py 的 predictions.csv)
1) 逐 seed: 配对 bootstrap(2000 次, 图像层面重采样)ΔMacro-F1 95% CI + McNemar 精确检验(对/错不一致对);
   跨 seed: ΔF1 的配对 t 检验(n=5)。
2) 误差结构: 按病理亚型 × 误判方向汇总 5 seeds 的错误数(baseline vs 最终网络)。
用法: python experiments/r6_significance.py [--a mobilenet_v3_large] [--b mbv3_fz16_dce4m]
输出: experiments/r6_significance.json
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import Counter

import numpy as np
from scipy.stats import binomtest, ttest_rel
from sklearn.metrics import f1_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]
SUB = re.compile(r"^test_(.+?)_S_")


def test_preds(root, model, seed):
    for f in sorted(glob.glob(os.path.join(REPO, root, f"{model}_seed{seed}", model, "eval_*", "results.json")), reverse=True):
        if os.path.basename(json.load(open(f))["test_dir"]) == "test":
            rows = list(csv.DictReader(open(os.path.join(os.path.dirname(f), "predictions.csv"))))
            return {os.path.basename(r["image_path"]): (C.index(r["true_label"]), C.index(r["pred_label"])) for r in rows}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="mobilenet_v3_large")
    ap.add_argument("--b", default="mbv3_fz16_dce4m")
    ap.add_argument("--root", default="weights_cle_ac_v1_r6")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    res, dF, err_a, err_b = {"per_seed": {}}, [], Counter(), Counter()
    for s in a.seeds:
        pa, pb = test_preds(a.root, a.a, s), test_preds(a.root, a.b, s)
        keys = sorted(pa)
        y = np.array([pa[k][0] for k in keys])
        ya = np.array([pa[k][1] for k in keys])
        yb = np.array([pb[k][1] for k in keys])
        f1a, f1b = f1_score(y, ya, average="macro"), f1_score(y, yb, average="macro")
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), len(y))
            boots.append(f1_score(y[idx], yb[idx], average="macro") - f1_score(y[idx], ya[idx], average="macro"))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        n01 = int(((ya == y) & (yb != y)).sum())          # a 对 b 错
        n10 = int(((ya != y) & (yb == y)).sum())          # a 错 b 对
        p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        res["per_seed"][s] = {"f1_a": round(f1a * 100, 2), "f1_b": round(f1b * 100, 2), "dF1": round((f1b - f1a) * 100, 2),
                              "dF1_CI95": [round(lo * 100, 2), round(hi * 100, 2)], "mcnemar_b_fixes": n10,
                              "mcnemar_b_breaks": n01, "mcnemar_p": float(p)}
        dF.append((f1b - f1a) * 100)
        for k, t, pa_, pb_ in zip(keys, y, ya, yb):
            st = SUB.match(k).group(1)[:5]
            if pa_ != t:
                err_a[f"{C[t][0]}:{st}->{C[pa_][0]}"] += 1
            if pb_ != t:
                err_b[f"{C[t][0]}:{st}->{C[pb_][0]}"] += 1
        print(f"seed{s}: F1 {f1a*100:.2f} -> {f1b*100:.2f}  ΔF1 {(f1b-f1a)*100:+.2f} [95% CI {lo*100:+.2f}, {hi*100:+.2f}]"
              f"  McNemar fixes/breaks {n10}/{n01} p={p:.2e}")
    t = ttest_rel(np.array(dF), np.zeros(len(dF)))
    res["paired_t"] = {"mean_dF1": round(float(np.mean(dF)), 3), "t": float(t.statistic), "p": float(t.pvalue), "n": len(dF)}
    print(f"across seeds: mean ΔF1 {np.mean(dF):+.2f}, paired t = {t.statistic:.2f}, p = {t.pvalue:.2e} (n={len(dF)})")
    keys = sorted(set(err_a) | set(err_b), key=lambda k: -(err_a[k] + err_b[k]))
    res["errors_5seeds"] = {k: {"a": err_a[k], "b": err_b[k]} for k in keys}
    print(f"\n{'error type (5 seeds total)':24s} {a.a[:14]:>14s} {a.b[:16]:>16s}")
    for k in keys:
        print(f"{k:24s} {err_a[k]:14d} {err_b[k]:16d}")
    print(f"{'TOTAL':24s} {sum(err_a.values()):14d} {sum(err_b.values()):16d}")
    json.dump(res, open(os.path.join(REPO, "experiments", "r6_significance.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
