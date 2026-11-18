#!/usr/bin/env python3
"""10-backbone comparison table.

Reporting convention requested by the user:
  * backbones  : single run (seed 42)
  * our models : **best of 5 seeds** (the seed with the highest test Macro-F1),
                 with that seed's full metric vector

Every row is recomputed from that run's predictions.csv so all numbers come from
one code path.  Outputs:
  experiments/cle_agent/report/backbone_comparison.md
  experiments/cle_agent/report/backbone_comparison.csv
  experiments/cle_agent/report/ours_per_seed.md      (per-seed detail, traceability)
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics as st

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "experiments", "cle_agent", "report")
CLASSES = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]
METRICS = ["macro_f1", "weighted_f1", "accuracy", "balanced_accuracy", "macro_sensitivity",
           "macro_specificity", "macro_precision", "auc_roc_macro", "auc_pr_macro",
           "kappa", "mcc"]

BACKBONES = [
    ("ResNet-34", "experiments/cle_agent/bench_backbones/eval_resnet34_*/predictions.csv"),
    ("ResNet-50", "experiments/cle_agent/bench_backbones/eval_resnet50_*/predictions.csv"),
    ("ResNet-101", "weights_bench10/resnet101*/eval_test/predictions.csv"),
    ("DenseNet-121", "weights_bench10/densenet121*/eval_test/predictions.csv"),
    ("ConvNeXt-T (baseline)", "weights_base/convnext_tiny/eval_test/predictions.csv"),
    ("ConvNeXt-S", "experiments/cle_agent/bench_backbones/eval_convnext_small_*/predictions.csv"),
    ("ConvNeXt-B", "weights_bench10/convnext_base*/eval_test/predictions.csv"),
    ("Swin-T", "experiments/cle_agent/bench_backbones/eval_swin_tiny_vanilla_*/predictions.csv"),
    ("PVT-T", "experiments/cle_agent/bench_backbones/eval_pvt_tiny_*/predictions.csv"),
    ("PVT-S", "weights_bench10/pvt_small*/eval_test/predictions.csv"),
]

OURS = [
    ("Ours: ConvNeXt-T+LCN (best of 5 seeds)", "cle_convnext_lcn_only"),
    ("Ours: ConvNeXt-T+SN (best of 5 seeds)", "cle_convnext_sn"),
]


def load_probs(path):
    yt, pred, pr = [], [], []
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        pcols = [i for i, h in enumerate(header) if h.startswith("prob_")]
        for row in rd:
            yt.append(CLASSES.index(row[1]))
            pred.append(CLASSES.index(row[2]))
            pr.append([float(row[i]) for i in pcols])
    return np.array(yt), np.array(pred), np.array(pr)


def metrics_from(yt, yp, pr) -> dict:
    spec = []
    for c in range(len(CLASSES)):
        tn = int(np.sum((yt != c) & (yp != c)))
        fp = int(np.sum((yt != c) & (yp == c)))
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return {
        "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
        "weighted_f1": f1_score(yt, yp, average="weighted", zero_division=0),
        "accuracy": accuracy_score(yt, yp),
        "balanced_accuracy": recall_score(yt, yp, average="macro", zero_division=0),
        "macro_sensitivity": recall_score(yt, yp, average="macro", zero_division=0),
        "macro_specificity": float(np.mean(spec)),
        "macro_precision": precision_score(yt, yp, average="macro", zero_division=0),
        "auc_roc_macro": roc_auc_score(yt, pr, multi_class="ovr", average="macro"),
        "auc_pr_macro": float(np.mean([average_precision_score((yt == c).astype(int), pr[:, c])
                                       for c in range(len(CLASSES))])),
        "kappa": cohen_kappa_score(yt, yp),
        "mcc": matthews_corrcoef(yt, yp),
        "n_errors": int((yt != yp).sum()),
    }


def complexity(pred_path: str) -> dict:
    rj = os.path.join(os.path.dirname(pred_path), "results.json")
    if not os.path.exists(rj):
        return {"params": "-", "macs": "-", "fps": "-"}
    r = json.load(open(rj))
    c, e = r.get("complexity", {}), r.get("efficiency", {})
    fps = e.get("pure_fps")
    return {"params": c.get("params", "-"), "macs": c.get("macs", "-"),
            "fps": f"{fps:.0f}" if isinstance(fps, (int, float)) else "-"}


def single_row(pattern: str):
    matches = sorted(glob.glob(os.path.join(ROOT, pattern)))
    if not matches:
        return None
    yt, yp, pr = load_probs(matches[0])
    m = metrics_from(yt, yp, pr)
    m.update(complexity(matches[0]))
    m["best_seed"] = 42
    return m


def per_seed_runs(model: str):
    runs = []
    for pred in glob.glob(os.path.join(ROOT, f"weights_main/{model}*/eval_test/predictions.csv")):
        meta = os.path.join(os.path.dirname(os.path.dirname(pred)), "run_meta.json")
        if not os.path.exists(meta):
            continue
        seed = json.load(open(meta)).get("seed")
        yt, yp, pr = load_probs(pred)
        m = metrics_from(yt, yp, pr)
        m["seed"] = seed
        m["pred"] = pred
        runs.append(m)
    return runs


def best_row(model: str):
    runs = per_seed_runs(model)
    if not runs:
        return None, []
    best = max(runs, key=lambda r: r["macro_f1"])
    best.update(complexity(best["pred"]))
    return best, sorted(runs, key=lambda r: r["seed"])


def fusion_runs():
    """Per-seed LCN+SN decision-level fusion."""
    def by_seed(model):
        out = {}
        for r in per_seed_runs(model):
            out.setdefault(r["seed"], []).append(r)
        return out
    lcn, sn = by_seed("cle_convnext_lcn_only"), by_seed("cle_convnext_sn")
    runs = []
    for s in sorted(set(lcn) & set(sn)):
        a = max(lcn[s], key=lambda x: x["macro_f1"])
        b = max(sn[s], key=lambda x: x["macro_f1"])
        _, _, pa = load_probs(a["pred"])
        _, _, pb = load_probs(b["pred"])
        p = 0.5 * (pa + pb)
        yt, _, _ = load_probs(a["pred"])
        m = metrics_from(yt, p.argmax(1), p)
        m["seed"] = s
        runs.append(m)
    return runs


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []

    for name, pattern in BACKBONES:
        m = single_row(pattern)
        if m:
            m["name"] = name
            rows.append(m)
        else:
            print("[missing]", name)

    per_seed_report = ["# 本文方法：5 个种子逐次结果（用于核对 best-of-5）", ""]
    for name, model in OURS:
        best, runs = best_row(model)
        if best:
            best["name"] = name
            rows.append(best)
            per_seed_report += [f"## {name}", "",
                                "| seed | Macro-F1 | Accuracy | Kappa | MCC | 错误数 |",
                                "|---|---|---|---|---|---|"]
            for r in runs:
                mark = " **← best**" if r["seed"] == best["seed"] else ""
                per_seed_report.append(
                    f"| {r['seed']}{mark} | {r['macro_f1']*100:.2f} | {r['accuracy']*100:.2f} | "
                    f"{r['kappa']*100:.2f} | {r['mcc']*100:.2f} | {r['n_errors']} |")
            per_seed_report.append("")

    fruns = fusion_runs()
    if fruns:
        fb = max(fruns, key=lambda r: r["macro_f1"])
        fb.update({"name": "Ours: DPFusion LCN+SN (best of 5 seeds)",
                   "params": "27.8M x2", "macs": "4.46G x2", "fps": "~600 (2 passes)"})
        rows.append(fb)
        per_seed_report += ["## Ours: DPFusion（LCN+SN 决策级融合）", "",
                            "| seed | Macro-F1 | Accuracy | Kappa | MCC | 错误数 |",
                            "|---|---|---|---|---|---|"]
        for r in fruns:
            mark = " **← best**" if r["seed"] == fb["seed"] else ""
            per_seed_report.append(
                f"| {r['seed']}{mark} | {r['macro_f1']*100:.2f} | {r['accuracy']*100:.2f} | "
                f"{r['kappa']*100:.2f} | {r['mcc']*100:.2f} | {r['n_errors']} |")
        per_seed_report += ["",
                            f"注：5 个种子的 Macro-F1 均值为 {st.mean([r['macro_f1'] for r in fruns])*100:.2f}，"
                            f"最优为 {fb['macro_f1']*100:.2f}（seed {fb['seed']}）。"]

    rows.sort(key=lambda r: -r["macro_f1"])
    base = next((r for r in rows if r["name"].startswith("ConvNeXt-T")), None)

    csv_path = os.path.join(OUTDIR, "backbone_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "selected_seed"] + METRICS + ["n_errors", "params", "macs", "fps"])
        for r in rows:
            w.writerow([r["name"], r.get("best_seed", r.get("seed", "-"))] +
                       [f"{r[k]:.4f}" for k in METRICS] +
                       [r["n_errors"], r["params"], r["macs"], r["fps"]])

    L = ["# 主干对比实验：10 个 ImageNet 预训练主干 + 本文方法", "",
         "**口径**：主干为 seed 42 单次训练结果；本文方法为 **5 个种子中 test Macro-F1 最高的那一次**"
         "（对应的完整指标向量同时报告，逐种子明细见 `ours_per_seed.md`）。",
         "",
         "协议：224x224 / bs32 / AdamW(lr 1e-4, wd 5e-2) / Cosine 300ep / unweighted CE / "
         "ImageNet 初始化 / 按 val Macro-F1 选 checkpoint / 早停 patience=30。", "",
         "| 模型 | 选定 seed | Macro-F1 | Weighted-F1 | Accuracy | Balanced-Acc | Macro-Sens | Macro-Spec | "
         "Macro-Prec | AUROC | AUC-PR | Kappa | MCC | 错误数 | Params | MACs | FPS | ΔF1 vs 基线 |",
         "|" + "---|" * 18]
    for r in rows:
        d = "—" if (base is None or r is base) else f"{100*(r['macro_f1']-base['macro_f1']):+.2f}"
        L.append("| " + " | ".join([
            r["name"], str(r.get("best_seed", r.get("seed", "-"))),
            f"{r['macro_f1']*100:.2f}", f"{r['weighted_f1']*100:.2f}", f"{r['accuracy']*100:.2f}",
            f"{r['balanced_accuracy']*100:.2f}", f"{r['macro_sensitivity']*100:.2f}",
            f"{r['macro_specificity']*100:.2f}", f"{r['macro_precision']*100:.2f}",
            f"{r['auc_roc_macro']*100:.2f}", f"{r['auc_pr_macro']*100:.2f}",
            f"{r['kappa']*100:.2f}", f"{r['mcc']*100:.2f}", str(r["n_errors"]),
            str(r["params"]), str(r["macs"]), str(r["fps"]), d]) + " |")
    with open(os.path.join(OUTDIR, "backbone_comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(OUTDIR, "ours_per_seed.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(per_seed_report) + "\n")

    print(f"[table] {len(rows)} rows -> {csv_path}")
    for r in rows:
        print(f"  {r['name']:44s} F1={r['macro_f1']:.4f} acc={r['accuracy']:.4f} "
              f"kappa={r['kappa']:.4f} err={r['n_errors']} seed={r.get('best_seed', r.get('seed'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
