#!/usr/bin/env python3
"""Evaluate every trained run on the TEST split and aggregate all 11 metrics.

Why this exists: the user's acceptance criterion is "any one of the evaluate.py
metrics improves by >= 1.5 points", so every run must be scored on test and all
metrics must be visible side by side.  Selection guidance still prefers val
(recorded in the training log), but the decision table is test-based by request.

All metrics are recomputed from `predictions.csv` (which holds per-class
softmax probabilities), so macro-precision / AUC-PR / kappa / MCC are available
even though `results.json` does not store all of them.

Usage:
  python scripts/eval_all.py                       # evaluate pending runs, print table
  python scripts/eval_all.py --rerun               # re-evaluate everything
  python scripts/eval_all.py --roots weights_main  # restrict scan roots
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_CANDIDATES = ["/opt/conda/envs/sunfuxiang/bin/python",
                 "/home/login/miniconda3/envs/sunfuxiang/bin/python"]
OUT_JSON = os.path.join(ROOT, "experiments", "cle_agent", "all_metrics.json")

METRIC_KEYS = ["macro_f1", "weighted_f1", "accuracy", "balanced_accuracy",
               "macro_sensitivity", "macro_specificity", "macro_precision",
               "auc_roc_macro", "auc_pr_macro", "kappa", "mcc"]

CLASS_NAMES = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]


def pick_python() -> str:
    for p in PY_CANDIDATES:
        if os.path.exists(p):
            return p
    return sys.executable


def compute_metrics_from_predictions(path: str) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score,
                                 f1_score, matthews_corrcoef, precision_score,
                                 recall_score, roc_auc_score)

    y_true, y_pred, probs = [], [], []
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        prob_cols = [i for i, h in enumerate(header) if h.startswith("prob_")]
        for row in rd:
            y_true.append(CLASS_NAMES.index(row[1]))
            y_pred.append(CLASS_NAMES.index(row[2]))
            probs.append([float(row[i]) for i in prob_cols])
    y_true = np.array(y_true); y_pred = np.array(y_pred); probs = np.array(probs)

    n_cls = len(CLASS_NAMES)
    per_cls_spec = []
    for c in range(n_cls):
        tn = int(np.sum((y_true != c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        per_cls_spec.append(tn / (tn + fp) if (tn + fp) else 0.0)

    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_sensitivity": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_specificity": float(np.mean(per_cls_spec)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "n_test": int(len(y_true)),
        "n_errors": int((y_true != y_pred).sum()),
    }
    try:
        out["auc_roc_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        out["auc_roc_macro"] = float("nan")
    aps = []
    for c in range(n_cls):
        aps.append(average_precision_score((y_true == c).astype(int), probs[:, c]))
    out["auc_pr_macro"] = float(np.mean(aps))
    return out


def discover(roots: list[str]) -> list[dict]:
    runs = []
    for root in roots:
        rp = os.path.join(ROOT, root)
        if not os.path.isdir(rp):
            continue
        for d in sorted(glob.glob(os.path.join(rp, "*"))):
            if not os.path.isdir(d):
                continue
            meta = os.path.join(d, "run_meta.json")
            ckpt = os.path.join(d, "model_best.pth")
            if not (os.path.exists(meta) and os.path.exists(ckpt)):
                continue
            with open(meta) as f:
                m = json.load(f)
            runs.append({"dir": d, "model": m.get("model"), "seed": m.get("seed"),
                         "val_macro_f1": None, "ckpt": ckpt})
    return runs


def read_val_from_log(model: str, seed: int) -> float | None:
    pats = [f"experiments/cle_agent/runs_*/{model}_seed{seed}.log",
            f"experiments/cle_agent/base_seed{seed}.log",
            f"experiments/cle_agent/smoke_convnext.log"]
    for pat in pats:
        for p in glob.glob(os.path.join(ROOT, pat)):
            best = None
            with open(p, errors="ignore") as f:
                for line in f:
                    if "Best val macro-F1:" in line:
                        try:
                            best = float(line.split("Best val macro-F1:")[1].split()[0])
                        except (IndexError, ValueError):
                            pass
            if best is not None:
                return best
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=["weights_main", "weights_bench", "weights_base"])
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    py = pick_python()
    runs = discover(args.roots)
    print(f"[scan] {len(runs)} run dirs under {args.roots}", flush=True)

    for r in runs:
        evaldir = os.path.join(r["dir"], "eval_test")
        rj = os.path.join(evaldir, "results.json")
        need = args.rerun or not os.path.exists(rj)
        if need and r["model"]:
            print(f"[eval] {os.path.basename(r['dir'])} ...", flush=True)
            subprocess.call([py, "evaluate.py", "--model", r["model"],
                             "--weights", r["ckpt"],
                             "--data-path", "./datasets/cle_ac_v1/test",
                             "--out-dir", evaldir],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pj = os.path.join(evaldir, "predictions.csv")
        if not os.path.exists(pj):
            r["status"] = "eval_failed"
            continue
        r.update(compute_metrics_from_predictions(pj))
        r["status"] = "ok"
        r.pop("ckpt", None)
        r["val_macro_f1"] = read_val_from_log(r["model"], r["seed"]) if r.get("seed") is not None else None

    # persist + print
    with open(OUT_JSON, "w") as f:
        json.dump(runs, f, indent=2, ensure_ascii=False)

    ok = [r for r in runs if r.get("status") == "ok"]
    ok.sort(key=lambda r: -r["macro_f1"])
    hdr = f"{'run':38s} {'seed':>4} {'valF1':>7} " + " ".join(f"{k[:9]:>9}" for k in METRIC_KEYS) + "  err"
    print("\n" + hdr)
    print("-" * len(hdr))
    base = next((r for r in ok if r["model"] == "convnext_tiny"), None)
    for r in ok:
        name = os.path.basename(r["dir"])
        v = r.get("val_macro_f1")
        cells = " ".join(f"{r[k]:9.4f}" if isinstance(r.get(k), (int, float)) and r.get(k) == r.get(k)
                         else f"{'NA':>9}" for k in METRIC_KEYS)
        print(f"{name:38s} {str(r.get('seed')):>4} {v if v is None else round(v,4)!s:>7} {cells}  {r['n_errors']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
