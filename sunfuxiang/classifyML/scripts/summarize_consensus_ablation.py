#!/usr/bin/env python3
"""Recompute consensus-ablation metrics from pinned predictions.csv files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from consensus_manifest import sha256_file  # noqa: E402


SUBSETS = ("Full", "R", "V", "C", "RV", "RC", "VC", "RVC")
SEEDS = (42, 43, 44)
OVERALL_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "macro_auroc_ovr",
)


def compute_metrics(y_true, y_pred, y_prob, class_names: list[str]) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    labels = np.arange(len(class_names))
    if y_prob.shape != (len(y_true), len(class_names)):
        raise ValueError(
            f"probability matrix shape {y_prob.shape} != ({len(y_true)}, {len(class_names)})"
        )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    try:
        macro_auroc = float(
            roc_auc_score(
                y_true,
                y_prob,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        macro_auroc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_auroc_ovr": macro_auroc,
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
    }


def load_predictions(path: Path, expected_rows: int | None = 671) -> tuple[dict, list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"image_path", "true_label", "pred_label"}
        if not required <= set(fieldnames):
            raise ValueError(f"{path}: missing columns {sorted(required-set(fieldnames))}")
        probability_fields = [name for name in fieldnames if name.startswith("prob_")]
        if len(probability_fields) < 2:
            raise ValueError(f"{path}: expected probability columns")
        class_names = [name.removeprefix("prob_") for name in probability_fields]
        class_to_index = {name: index for index, name in enumerate(class_names)}
        rows = list(reader)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {len(rows)}")
    image_paths = [row["image_path"] for row in rows]
    if len(set(image_paths)) != len(image_paths):
        raise ValueError(f"{path}: duplicate image_path rows")
    try:
        y_true = [class_to_index[row["true_label"]] for row in rows]
        y_pred = [class_to_index[row["pred_label"]] for row in rows]
    except KeyError as exc:
        raise ValueError(f"{path}: label not represented by probability columns: {exc}") from exc
    y_prob = [[float(row[name]) for name in probability_fields] for row in rows]
    return compute_metrics(y_true, y_pred, y_prob, class_names), class_names


def aggregate_seed_metrics(seed_metrics: list[dict]) -> dict:
    if not seed_metrics:
        return {}
    keys = [key for key in OVERALL_METRICS if all(key in row for row in seed_metrics)]
    if not keys:
        keys = sorted(
            key
            for key, value in seed_metrics[0].items()
            if isinstance(value, (int, float)) and all(key in row for row in seed_metrics)
        )
    result = {}
    for key in keys:
        values = np.asarray([row[key] for row in seed_metrics], dtype=float)
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n": int(len(values)),
            "ddof": 1,
            "formatted": (
                f"{np.mean(values):.6f} ± "
                f"{(np.std(values, ddof=1) if len(values)>1 else 0.0):.6f}"
            ),
        }
    return result


def aggregate_per_class(seed_metrics: list[dict], class_names: list[str]) -> dict:
    output = {}
    for class_name in class_names:
        output[class_name] = {}
        for metric in ("precision", "recall", "f1"):
            values = np.asarray(
                [row["per_class"][class_name][metric] for row in seed_metrics], dtype=float
            )
            output[class_name][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
                "ddof": 1,
            }
    return output


def load_legacy_map(path: Path) -> dict[tuple[str, int], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for subset in ("Full", "RVC"):
        for seed_text, entry in payload["runs"][subset].items():
            predictions = Path(entry["predictions"])
            if not predictions.is_file():
                raise FileNotFoundError(predictions)
            if entry.get("sha256") and sha256_file(predictions) != entry["sha256"]:
                raise ValueError(f"legacy prediction hash mismatch: {predictions}")
            result[(subset, int(seed_text))] = predictions
    return result


def expected_paths(legacy_map: Path, new_root: Path) -> dict[tuple[str, int], Path]:
    paths = load_legacy_map(legacy_map)
    for subset in ("R", "V", "C", "RV", "RC", "VC"):
        for seed in SEEDS:
            paths[(subset, seed)] = (
                new_root
                / subset
                / f"seed{seed}"
                / "resnet34"
                / "eval_fixed_test"
                / "predictions.csv"
            )
    return paths


def run(args: argparse.Namespace) -> int:
    paths = expected_paths(args.legacy_map, args.new_root)
    found, missing = {}, []
    reference_classes = None
    for subset in SUBSETS:
        for seed in SEEDS:
            path = paths[(subset, seed)]
            if not path.is_file():
                missing.append({"subset": subset, "seed": seed, "path": str(path)})
                continue
            metrics, class_names = load_predictions(path, expected_rows=args.expected_rows)
            if reference_classes is None:
                reference_classes = class_names
            elif class_names != reference_classes:
                raise ValueError(f"class order mismatch in {path}: {class_names}")
            found[(subset, seed)] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "metrics": metrics,
            }

    report = {
        "primary_metric": "macro_f1",
        "ddof": 1,
        "expected_runs": len(SUBSETS) * len(SEEDS),
        "found_runs": len(found),
        "missing_runs": missing,
        "class_names": reference_classes,
        "runs": {},
        "summary": {},
    }
    for (subset, seed), entry in found.items():
        report["runs"].setdefault(subset, {})[str(seed)] = entry
    for subset in SUBSETS:
        available = [
            found[(subset, seed)]["metrics"]
            for seed in SEEDS
            if (subset, seed) in found
        ]
        if len(available) == len(SEEDS):
            report["summary"][subset] = {
                "overall": aggregate_seed_metrics(available),
                "per_class": aggregate_per_class(available, reference_classes),
            }

    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    if args.dry_run:
        if len(found) < 6 or any(key not in found for key in [("Full", 42), ("RVC", 42)]):
            raise RuntimeError("dry-run did not validate all six pinned legacy runs")
        return 0
    if missing:
        raise RuntimeError(f"cannot summarize: {len(missing)} prediction files are missing")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "CONSENSUS_ABLATION_METRICS.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (args.out_dir / "CONSENSUS_ABLATION_SEED_METRICS.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["subset", "seed", *OVERALL_METRICS, "predictions", "sha256"])
        for subset in SUBSETS:
            for seed in SEEDS:
                entry = found[(subset, seed)]
                metrics = entry["metrics"]
                writer.writerow(
                    [subset, seed, *[f"{metrics[key]:.9f}" for key in OVERALL_METRICS], entry["path"], entry["sha256"]]
                )
    with (args.out_dir / "CONSENSUS_ABLATION_SUMMARY.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["subset", *[f"{key}_mean" for key in OVERALL_METRICS], *[f"{key}_std" for key in OVERALL_METRICS]])
        for subset in SUBSETS:
            overall = report["summary"][subset]["overall"]
            writer.writerow(
                [subset, *[f"{overall[key]['mean']:.9f}" for key in OVERALL_METRICS], *[f"{overall[key]['std']:.9f}" for key in OVERALL_METRICS]]
            )
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-map",
        type=Path,
        default=REPO / "consensus_ablation_audit/LEGACY_PREDICTION_MAP.json",
    )
    parser.add_argument(
        "--new-root", type=Path, default=REPO / "weights_consensus_ablation"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO / "consensus_ablation_results"
    )
    parser.add_argument("--expected-rows", type=int, default=671)
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(make_parser().parse_args()))
