#!/usr/bin/env python3
"""修复版消融表聚合 (Step 4)。

口径铁律: 由原始 results.json 的**未舍入** macro_f1 聚合,
mean / sample std (ddof=1) 计算完毕后才舍入到 4 位小数。
旧 aggregate.py 先 round(4) 再聚合会引入 ±0.0001 级偏差, 本脚本不沿用。

用法:
    python experiments/aggregate_fixed.py --root weights_cle_fixed_v1
    python experiments/aggregate_fixed.py --root weights_cle_fixed_v1 \
        --extra-root /home/login/data/cle/auto/ZCode/Classification/weights_cle_ac_v1
"""
import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSES = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]


def collect(root, model_filter=None):
    """{model: {seed: {...}}} —— 每个 run 取 test 集最新一次 eval。"""
    if not os.path.isabs(root):
        root = os.path.join(REPO, root)
    out = defaultdict(dict)
    for run_dir in sorted(glob.glob(os.path.join(root, "*_seed*"))):
        base = os.path.basename(run_dir)
        model, seed = base.rsplit("_seed", 1)
        if model_filter and model not in model_filter:
            continue
        if not seed.isdigit():
            continue
        cands = sorted(glob.glob(os.path.join(run_dir, "*", "eval_*", "results.json")))
        # 只保留 test 集评估, 取时间戳最新的一份
        tests = []
        for f in cands:
            try:
                r = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            if "test" in r.get("test_dir", ""):
                tests.append((os.path.basename(os.path.dirname(f)), r, f))
        if not tests:
            continue
        tests.sort(key=lambda t: t[0])
        stamp, r, path = tests[-1]
        per = r.get("per_class", {})
        rec = {
            "f1": float(r["macro_f1"]),
            "acc": float(r.get("accuracy", float("nan"))),
            "auroc": float(r.get("auc_roc_macro", float("nan"))),
            "n_total": r.get("n_total"),
            "params_raw": (r.get("complexity") or {}).get("params_raw"),
            "path": os.path.relpath(path, REPO) if path.startswith(REPO) else path,
        }
        for c in CLASSES:
            rec["sens_" + c] = float(per.get(c, {}).get("sensitivity", float("nan")))
        cm = r.get("confusion_matrix")
        if cm and len(cm) >= 2:
            # 0=Benign 1=Malignant, 两者互误判
            rec["bm_err"] = int(cm[0][1] + cm[1][0])
            rec["err_total"] = int(sum(sum(row) for row in cm) -
                                   sum(cm[i][i] for i in range(len(cm))))
        out[model][int(seed)] = rec
    return out


def agg(vals):
    vals = [v for v in vals if v == v]  # drop NaN
    if not vals:
        return None
    mean = st.mean(vals)
    std = st.stdev(vals) if len(vals) > 1 else float("nan")
    return {"n": len(vals), "mean": mean, "std": std,
            "var": std ** 2 if std == std else float("nan"),
            "min": min(vals), "max": max(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="weights_cle_fixed_v1")
    ap.add_argument("--extra-root", action="append", default=[],
                    help="基线/对照权重根(可多次), 例如 ZCode 的 convnext_tiny")
    ap.add_argument("--baseline", default="convnext_tiny")
    ap.add_argument("--out", default="experiments/SUMMARY_CLE_FIXED_V1.md")
    args = ap.parse_args()

    data = collect(args.root)
    for er in args.extra_root:
        extra = collect(er, model_filter={args.baseline})
        for m, seeds in extra.items():
            data[m].update(seeds)

    if not data:
        print("no results yet")
        return

    order = ["convnext_tiny", "convnext_egb", "convnext_esa", "convnext_scg",
             "convnext_hsc", "convnext_egb_scg", "convnext_scg_hsc",
             "convnext_egb_scg_hsc", "convnext_esa_scg_hsc"]
    models = [m for m in order if m in data] + [m for m in sorted(data) if m not in order]

    stats = {}
    for m in models:
        recs = data[m]
        stats[m] = {
            "f1": agg([r["f1"] for r in recs.values()]),
            "acc": agg([r["acc"] for r in recs.values()]),
            "auroc": agg([r["auroc"] for r in recs.values()]),
            "sens_B": agg([r["sens_" + CLASSES[0]] for r in recs.values()]),
            "sens_M": agg([r["sens_" + CLASSES[1]] for r in recs.values()]),
            "sens_N": agg([r["sens_" + CLASSES[2]] for r in recs.values()]),
            "bm_err": agg([r.get("bm_err", float("nan")) for r in recs.values()]),
            "err": agg([r.get("err_total", float("nan")) for r in recs.values()]),
            "seeds": sorted(recs),
            "best": max(recs.items(), key=lambda kv: kv[1]["f1"]),
            "params": next((r["params_raw"] for r in recs.values() if r["params_raw"]), None),
        }

    def fmt(a, nd=4):
        if not a:
            return "-"
        if a["std"] != a["std"]:
            return f"{a['mean']:.{nd}f}"
        return f"{a['mean']:.{nd}f}±{a['std']:.{nd}f}"

    lines = []
    lines.append("| 配置 | n | seeds | Macro-F1 mean±std | Acc | AUROC | Sens-B | Sens-M | Sens-N | B↔M 误判 | 总错误 | params |")
    lines.append("|---|---:|---|---|---|---|---|---|---|---:|---:|---:|")
    for m in models:
        s = stats[m]
        b = s["best"]
        lines.append(
            f"| {m} | {s['f1']['n']} | {','.join(map(str, s['seeds']))} | {fmt(s['f1'])} | "
            f"{fmt(s['acc'])} | {fmt(s['auroc'])} | {fmt(s['sens_B'])} | {fmt(s['sens_M'])} | "
            f"{fmt(s['sens_N'])} | {fmt(s['bm_err'],2)} | {fmt(s['err'],2)} | "
            f"{s['params'] if s['params'] else '-'} |")

    lines.append("")
    lines.append("| 配置 | best F1 | best seed | 相对 baseline 均值 | 相对 baseline 标准差 |")
    lines.append("|---|---:|---:|---|---|")
    base = stats.get(args.baseline)
    for m in models:
        s = stats[m]
        b = s["best"]
        if base and base["f1"] and m != args.baseline:
            dm = (s["f1"]["mean"] - base["f1"]["mean"]) * 100
            ds = (s["f1"]["std"] / base["f1"]["std"] - 1) * 100
            dv = (s["f1"]["var"] / base["f1"]["var"] - 1) * 100
            rel = f"{dm:+.2f}pp"
            relstd = f"{ds:+.2f}% (方差 {dv:+.2f}%)"
        else:
            rel, relstd = "-", "-"
        lines.append(f"| {m} | {b[1]['f1']:.4f} | {b[0]} | {rel} | {relstd} |")

    lines.append("")
    lines.append("## 逐 run 原始 macro-F1 (未舍入来自 results.json)")
    lines.append("")
    for m in models:
        cells = "  ".join(f"s{seed}={rec['f1']:.6f}" for seed, rec in sorted(data[m].items()))
        lines.append(f"- **{m}**: {cells}")

    lines.append("")
    lines.append("## 统计口径")
    lines.append("")
    lines.append("- 输入: 各 `eval_*/results.json` 的 `macro_f1` 原始 float, 未做任何预先舍入;")
    lines.append("- 聚合: mean 与 sample std (ddof=1) 均在原始精度下计算, 最后才舍入 4 位;")
    lines.append("- 单 run 配置不计算 std;")
    lines.append("- `params` 取自 evaluate.py 的 `complexity.params_raw`(THOP 口径, 会漏计 layer_scale/自定义 LN)。")

    doc = "\n".join(lines)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 修复版消融表 (weights_cle_fixed_v1, test 集)\n\n" + doc + "\n")
    print(doc)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
