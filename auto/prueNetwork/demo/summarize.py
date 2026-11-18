"""汇总 demo/results/ 下各项目 D4 对称化涨点 → Markdown 表格(README 用)。"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = ("macro_f1", "accuracy", "weighted_f1", "balanced_accuracy",
           "auc_roc_macro", "macro_sensitivity", "macro_specificity",
           "kappa", "mcc")


def main():
    rows_all = {}
    for proj_dir in sorted((HERE / "results").iterdir()):
        if not proj_dir.is_dir():
            continue
        rows = [json.loads(f.read_text()) for f in sorted(proj_dir.glob("seed*.json"))]
        if rows:
            rows_all[proj_dir.name] = rows
    if not rows_all:
        sys.exit("no results yet")

    lines = []
    for name, rows in rows_all.items():
        seeds = [r["seed"] for r in rows]
        lines.append(f"\n### {name}（`{rows[0]['model']}`，seeds={seeds}）\n")
        lines.append("| 指标 | 单视图均值 | D4均值 | Δ(均值) | 逐seed Δ |")
        lines.append("|---|---:|---:|---:|---|")
        pos = 0
        for k in METRICS:
            s = [r["single"][k] for r in rows]
            d = [r["d4"][k] for r in rows]
            diff = [di - si for si, di in zip(s, d)]
            dm = statistics.mean(diff)
            pos += dm > 0
            lines.append(f"| {k} | {statistics.mean(s):.4f} | {statistics.mean(d):.4f} "
                         f"| {dm:+.4f} | {' '.join(f'{x:+.4f}' for x in diff)} |")
        lines.append(f"\n→ **{pos}/9 项指标均值上涨点**（macro_f1 Δ="
                     f"{statistics.mean(r['d4']['macro_f1']-r['single']['macro_f1'] for r in rows):+.4f}）")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
