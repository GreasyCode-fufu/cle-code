#!/usr/bin/env python3
"""Create the registered final consensus-ablation tables and narrative report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ORDER = ("Full", "R", "V", "C", "RV", "RC", "VC", "RVC")
SEEDS = (42, 43, 44)
SIZES = {"Full": 4977, "R": 3500, "V": 3474, "C": 3749,
         "RV": 3103, "RC": 3141, "VC": 3115, "RVC": 2880}
COUNTS = {
    "Full": (1578, 2136, 1263), "R": (1405, 1458, 637),
    "V": (1420, 1415, 639), "C": (1431, 1476, 842),
    "RV": (1346, 1255, 502), "RC": (1349, 1234, 558),
    "VC": (1345, 1210, 560), "RVC": (1303, 1114, 463),
}
METRICS = ("macro_f1", "accuracy", "macro_recall", "macro_precision",
           "weighted_f1", "macro_auroc_ovr")


def fmt(mean: float, std: float) -> str:
    return f"{mean:.6f} ± {std:.6f}"


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    if report["missing_runs"] or report["found_runs"] != 24:
        raise RuntimeError("finalization requires all 24 Full/new/RVC seed predictions")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    cm_dir = out / "CONFUSION_MATRICES"
    cm_dir.mkdir(exist_ok=True)

    overall_rows = []
    for subset in ORDER:
        agg = report["summary"][subset]["overall"]
        overall_rows.append([
            subset, SIZES[subset], SIZES[subset] / SIZES["Full"],
            *[agg[key]["mean"] for key in METRICS],
            *[agg[key]["std"] for key in METRICS],
            *[fmt(agg[key]["mean"], agg[key]["std"]) for key in METRICS],
        ])
    write_csv(
        out / "CONSENSUS_ABLATION_OVERALL.csv",
        ["subset", "train_size", "retention_rate",
         *[f"{key}_mean" for key in METRICS], *[f"{key}_std" for key in METRICS],
         *[f"{key}_mean_std" for key in METRICS]],
        overall_rows,
    )

    per_seed_rows, per_class_rows = [], []
    class_names = report["class_names"]
    for subset in ORDER:
        for seed in SEEDS:
            run = report["runs"][subset][str(seed)]
            metric = run["metrics"]
            per_seed_rows.append([
                subset, seed, SIZES[subset], *[metric[key] for key in METRICS],
                run["path"], run["sha256"],
            ])
            matrix = metric["confusion_matrix"]
            write_csv(
                cm_dir / f"{subset}_seed{seed}.csv",
                ["true\\pred", *class_names],
                [[class_names[i], *matrix[i]] for i in range(len(class_names))],
            )
        class_agg = report["summary"][subset]["per_class"]
        for class_name in class_names:
            per_class_rows.append([
                subset, class_name,
                *[class_agg[class_name][key]["mean"] for key in ("precision", "recall", "f1")],
                *[class_agg[class_name][key]["std"] for key in ("precision", "recall", "f1")],
            ])
    write_csv(
        out / "CONSENSUS_ABLATION_PER_SEED.csv",
        ["subset", "seed", "train_size", *METRICS, "predictions", "sha256"],
        per_seed_rows,
    )
    write_csv(
        out / "CONSENSUS_ABLATION_PER_CLASS.csv",
        ["subset", "class_name", "precision_mean", "recall_mean", "f1_mean",
         "precision_std", "recall_std", "f1_std"],
        per_class_rows,
    )

    means = {s: report["summary"][s]["overall"]["macro_f1"]["mean"] for s in ORDER}
    stds = {s: report["summary"][s]["overall"]["macro_f1"]["std"] for s in ORDER}
    comparisons = [
        ("R - Full", "R", "Full"), ("V - Full", "V", "Full"),
        ("C - Full", "C", "Full"), ("RV - R", "RV", "R"),
        ("RV - V", "RV", "V"), ("RC - R", "RC", "R"),
        ("RC - C", "RC", "C"), ("VC - V", "VC", "V"),
        ("VC - C", "VC", "C"), ("RVC - RV", "RVC", "RV"),
        ("RVC - RC", "RVC", "RC"), ("RVC - VC", "RVC", "VC"),
    ]
    deltas = [(label, means[a] - means[b]) for label, a, b in comparisons]
    singles_above_full = [s for s in ("R", "V", "C") if means[s] > means["Full"]]
    pair_tests = [("RV", "R"), ("RV", "V"), ("RC", "R"),
                  ("RC", "C"), ("VC", "V"), ("VC", "C")]
    pair_wins = sum(means[a] > means[b] for a, b in pair_tests)
    triple_wins = sum(means["RVC"] > means[p] for p in ("RV", "RC", "VC"))
    best = max(ORDER, key=means.get)
    malignant = "Malignant_lesions"
    malignant_lines = []
    for subset in ORDER:
        pc = report["summary"][subset]["per_class"][malignant]
        malignant_lines.append(
            f"| {subset} | {fmt(pc['recall']['mean'], pc['recall']['std'])} | "
            f"{fmt(pc['f1']['mean'], pc['f1']['std'])} |"
        )

    monotonic_size = all(
        (SIZES[a] < SIZES[b]) == (means[a] <= means[b])
        for i, a in enumerate(ORDER) for b in ORDER[i + 1:] if SIZES[a] != SIZES[b]
    )
    consensus_trend = (
        len(singles_above_full) >= 2 and pair_wins >= 4 and triple_wins >= 2
    )
    rvc_strict = means["RVC"] > max(means[p] for p in ("RV", "RC", "VC"))

    lines = [
        "# CLE 共识消融结果分析", "",
        "所有指标均从锁定的 `predictions.csv` 统一重算；每组为三个训练 seed 的 mean ± sample std（ddof=1）。", "",
        "## 最终八组表", "",
        "| Subset | N | Macro-F1 | Accuracy | Macro Recall | Macro Precision | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in ORDER:
        agg = report["summary"][subset]["overall"]
        lines.append(
            f"| {subset} | {SIZES[subset]} | {fmt(agg['macro_f1']['mean'], agg['macro_f1']['std'])} | "
            f"{fmt(agg['accuracy']['mean'], agg['accuracy']['std'])} | "
            f"{fmt(agg['macro_recall']['mean'], agg['macro_recall']['std'])} | "
            f"{fmt(agg['macro_precision']['mean'], agg['macro_precision']['std'])} | "
            f"{fmt(agg['macro_auroc_ovr']['mean'], agg['macro_auroc_ovr']['std'])} |"
        )
    lines += ["", "## 递进关系（Macro-F1 mean 差值）", "",
              "| Comparison | Δ |", "|---|---:|"]
    lines += [f"| {label} | {delta:+.6f} |" for label, delta in deltas]
    lines += [
        "", "## 训练集大小与类别组成", "",
        "| Subset | Train size | Retention | Normal | Benign | Malignant | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in ORDER:
        normal, benign, mal = COUNTS[subset]
        lines.append(
            f"| {subset} | {SIZES[subset]} | {SIZES[subset]/SIZES['Full']:.3%} | "
            f"{normal} | {benign} | {mal} | {fmt(means[subset], stds[subset])} |"
        )
    lines += [
        "", "训练集大小与 Macro-F1 " + ("呈单调对应。" if monotonic_size else "不呈单调对应；这只是描述性结果，不能据此作因果结论。"),
        "", "## Malignant 类", "",
        "| Subset | Recall | F1 |", "|---|---:|---:|", *malignant_lines,
        "", "## 对预注册问题的回答", "",
        "1. 八组 Macro-F1 已列于最终表。",
        f"2. Single consensus 中高于 Full 的为：{', '.join(singles_above_full) if singles_above_full else '无'}。",
        f"3. Pairwise 相对其对应 Single 的 6 个比较中有 {pair_wins} 个提高，因此" + ("总体支持提高。" if pair_wins >= 4 else "不支持稳定提高。"),
        f"4. RVC 高于三个 Pairwise 中的 {triple_wins} 个；" + ("RVC 严格高于全部 Pairwise。" if rvc_strict else "不能称为严格优于全部 Pairwise。"),
        f"5. 按三 seed 平均 Macro-F1，最优 subset 是 {best}（{means[best]:.6f}）。",
        f"6. Full → Single → Pairwise → Triple 的总体提升趋势" + ("存在，但仍有局部例外。" if consensus_trend else "不够一致，存在关键反例。"),
        "7. 类别组成随筛选明显变化，尤其 Malignant 保留比例下降，可能影响指标解释；这里只作描述，不作因果归因。",
        "8. Malignant recall/F1 已在专表中逐组报告。",
        "9. " + ("RVC 的均值高于全部单/双共识组，但是否足以作强论断仍应结合 seed 波动与等规模对照。" if rvc_strict and all(means['RVC'] > means[s] for s in ('R','V','C')) else "RVC 未同时严格胜过全部单/双共识组，不足以支持‘三模型共识必然优于单/双模型’。"),
        "10. 建议继续运行预注册的 Random-2880 等规模对照，以区分共识筛选效应与训练集规模/类别组成效应；本阶段未运行 Random-2880。",
        "", "## 结论边界", "",
        "上述比较基于三个训练 seed 的描述性均值与样本标准差；未进行显著性检验，也不把 test 用于模型选择或超参数调整。", "",
    ]
    (out / "CONSENSUS_ABLATION_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"finalized results in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
