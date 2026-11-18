#!/usr/bin/env python3
"""汇总实验结果: 扫描 weights_cle_ac_v1/*/eval_*/results.json, 输出 test 集全指标对比表。

用法:
    python experiments/aggregate.py              # 全部 run
    python experiments/aggregate.py n1b1 n1b2    # 只看指定 batch (按 weights 目录 mtime 过滤不便,
                                                 # 故直接按模型名匹配, batch 维度从 results csv 取)
输出:
    experiments/summary.md  +  stdout 表格
"""
import glob
import json
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WROOT = os.path.join(REPO, "weights_cle_ac_v1")
OUT = os.path.join(REPO, "experiments", "summary.md")

# evaluate.py results.json 中可用的指标键(存在才取)
METRIC_KEYS = [
    ("accuracy", "Acc"),
    ("macro_f1", "Macro-F1"),
    ("weighted_f1", "Weighted-F1"),
    ("balanced_accuracy", "Bal-Acc"),
    ("macro_sensitivity", "Macro-Sens"),
    ("macro_specificity", "Macro-Spec"),
    ("macro_precision", "Macro-Prec"),
    ("auc_roc_macro", "AUROC"),
    ("auc_pr_macro", "AUC-PR"),
    ("cohen_kappa", "Kappa"),
    ("mcc", "MCC"),
]


def find_test_results():
    """{model: {seed: {metric: value}}} —— 只取 test 集结果。"""
    out = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(WROOT, "*_seed*", "*", "eval_*", "results.json"))):
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        td = r.get("test_dir", "")
        if "test" not in td:
            continue
        # 路径: weights_cle_ac_v1/<model>_seed<seed>/<model>/eval_*/
        parts = f.replace("\\", "/").split("/")
        run = [p for p in parts if "_seed" in p]
        if not run:
            continue
        name = run[0]
        model, seed = name.rsplit("_seed", 1)
        m = {}
        for k, _ in METRIC_KEYS:
            v = r.get(k)
            if isinstance(v, (int, float)):
                m[k] = round(float(v), 4)
        # per-class sensitivity
        per = r.get("per_class", {})
        for cls, short in [("Benign_lesions", "sens_benign"),
                           ("Malignant_lesions", "sens_mal"),
                           ("body_of_stomach_normal", "sens_normal")]:
            v = per.get(cls, {}).get("sensitivity")
            if isinstance(v, (int, float)):
                m[short] = round(float(v), 4)
        out[model][int(seed)] = m
    return out


def main():
    data = find_test_results()
    if not data:
        print("no results yet")
        return
    keys = [k for k, _ in METRIC_KEYS] + ["sens_benign", "sens_mal", "sens_normal"]
    header = ["model", "seed"] + [lbl for _, lbl in METRIC_KEYS] + ["Sens-B", "Sens-M", "Sens-N"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    best_rows = []
    for model in sorted(data):
        seeds = data[model]
        best_f1, best_seed = -1, None
        for seed in sorted(seeds):
            m = seeds[seed]
            row = [model, str(seed)] + [f"{m.get(k, float('nan')):.4f}" if k in m else "-" for k in keys]
            lines.append("| " + " | ".join(row) + " |")
            if m.get("macro_f1", -1) > best_f1:
                best_f1, best_seed = m["macro_f1"], seed
        if best_seed is not None:
            m = seeds[best_seed]
            row = [f"**{model} (best)**", str(best_seed)] + \
                  [f"**{m[k]:.4f}**" if k in m else "-" for k in keys]
            lines.append("| " + " | ".join(row) + " |")
            best_rows.append((model, best_seed, m))
    table = "\n".join(lines)

    # 相对 convnext_tiny best 的 Macro-F1 提升
    base = next((m for md, _, m in best_rows if md == "convnext_tiny"), None)
    delta_lines = []
    if base:
        bf1 = base.get("macro_f1")
        for model, seed, m in best_rows:
            d = (m.get("macro_f1", 0) - bf1) * 100
            delta_lines.append(f"- {model} (seed{seed}): Macro-F1 {m.get('macro_f1'):.4f}, "
                               f"vs baseline {d:+.2f}pp")
    delta = "\n".join(delta_lines)

    doc = f"# CLE 分类实验汇总 (test 集)\n\n{table}\n\n## 相对 convnext_tiny 最优种的提升\n\n{delta}\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(doc)
    print(doc)


if __name__ == "__main__":
    main()
