#!/usr/bin/env python3
"""汇总指定模型的 test 全指标(kappa/mcc/f1/...) — 从 weights 目录读 eval results.json。
用法: python experiments/collect_metrics.py convnext_dph 42 43 [--val]
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.environ.get("WEIGHTS_ROOT", "weights_cle_ac_v1")

KEYS = ["macro_f1", "accuracy", "weighted_f1", "balanced_accuracy", "kappa",
        "mcc", "macro_sensitivity", "macro_specificity",
        "auc_roc_macro"]


def load(model, seed, want_val=False):
    pats = os.path.join(REPO, W, f"{model}_seed{seed}", model, "eval_*", "results.json")
    out = None
    for f in sorted(glob.glob(pats)):
        r = json.load(open(f))
        is_val = "val" in r.get("test_dir", "")
        if is_val == want_val:
            out = r
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_val = "--val" in sys.argv
    model, seeds = args[0], [int(s) for s in args[1:]]
    rows = []
    for s in seeds:
        r = load(model, s, want_val)
        if not r:
            print(f"[miss] {model} seed{s}")
            continue
        rows.append({k: r.get(k, float("nan")) * 100 for k in KEYS})
        per = r.get("per_class", {})
        m = per.get("Malignant_lesions", {})
        rows[-1]["sens_M"] = m.get("sensitivity", float("nan")) * 100
    if not rows:
        return
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    std = {k: float(np.std([r[k] for r in rows])) for k in rows[0]}
    print(f"\n== {model} ({'val' if want_val else 'test'}) n={len(rows)} seeds={seeds}")
    print("  ".join(f"{k}={mean[k]:.2f}±{std[k]:.2f}" for k in
                    ["macro_f1", "kappa", "mcc", "balanced_accuracy", "sens_M"]))
    for k in KEYS + ["sens_M"]:
        print(f"  {k:20s} {mean[k]:7.2f} ±{std[k]:.2f}   [{', '.join(f'{r[k]:.2f}' for r in rows)}]")


if __name__ == "__main__":
    main()
