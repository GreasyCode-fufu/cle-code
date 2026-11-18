#!/usr/bin/env python3
"""Strict completeness validator for one consensus-ablation run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


EXPECTED_COUNTS = {
    "R": (3500, 1405, 1458, 637),
    "V": (3474, 1420, 1415, 639),
    "C": (3749, 1431, 1476, 842),
    "RV": (3103, 1346, 1255, 502),
    "RC": (3141, 1349, 1234, 558),
    "VC": (3115, 1345, 1210, 560),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--subset", choices=EXPECTED_COUNTS, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--require-eval", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    subset, seed = args.subset, args.seed
    manifest = repo / "manifests" / "consensus_ablation" / f"{subset}.csv"
    metadata = json.loads(
        (repo / "manifests" / "consensus_ablation" / "MANIFEST_METADATA.json").read_text(
            encoding="utf-8"
        )
    )
    registered = metadata["subsets"][subset]
    require(manifest.is_file(), f"missing manifest: {manifest}")
    require(sha256(manifest) == registered["sha256"], "manifest hash mismatch")

    run_dir = repo / "weights_consensus_ablation" / subset / f"seed{seed}" / "resnet34"
    best = run_dir / f"resnet34_{subset}_seed{seed}_best.pth"
    train_log = run_dir / f"train_log_resnet34_{subset}_seed{seed}.csv"
    run_meta = run_dir / "run_meta.json"
    require(best.is_file() and best.stat().st_size > 1_000_000, "best checkpoint missing/invalid")
    require(train_log.is_file() and train_log.stat().st_size > 0, "train log missing/empty")
    require(run_meta.is_file(), "run_meta missing")

    with train_log.open("r", encoding="utf-8-sig", newline="") as handle:
        train_rows = list(csv.DictReader(handle))
    require(len(train_rows) >= 1, "train log has no epoch rows")
    required_train_cols = {
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr"
    }
    require(required_train_cols <= set(train_rows[0]), "train log columns invalid")

    meta = json.loads(run_meta.read_text(encoding="utf-8"))
    total, normal, benign, malignant = EXPECTED_COUNTS[subset]
    exact = {
        "model": "resnet34",
        "num_classes": 3,
        "img_size": 224,
        "seed": seed,
        "subset": subset,
        "manifest_sha256": registered["sha256"],
        "actual_train_count": total,
        "normal_count": normal,
        "benign_count": benign,
        "malignant_count": malignant,
        "batch_size": 32,
        "max_epochs": 300,
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.05,
        "scheduler": "CosineAnnealingLR(T_max=300, eta_min=0)",
        "label_smoothing": 0.0,
        "early_stop_patience": 30,
        "amp_enabled": True,
        "best_checkpoint_metric": "val_macro_f1",
        "pretrained_sha256": "333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef",
    }
    for key, expected in exact.items():
        require(meta.get(key) == expected, f"run_meta {key}: {meta.get(key)!r} != {expected!r}")
    require(isinstance(meta.get("code_hash"), dict) and meta["code_hash"], "code_hash missing")
    require(meta.get("timestamp"), "timestamp missing")

    if args.require_eval:
        eval_dir = run_dir / "eval_fixed_test"
        predictions = eval_dir / "predictions.csv"
        results = eval_dir / "results.json"
        require(predictions.is_file(), "predictions.csv missing")
        require(results.is_file(), "results.json missing")
        with predictions.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == 671, f"predictions row count {len(rows)} != 671")
        require(len({row["image_path"] for row in rows}) == 671, "duplicate image_path")
        prob_cols = [name for name in rows[0] if name.startswith("prob_")]
        require(len(prob_cols) == 3, f"expected 3 probability columns, got {prob_cols}")
        for index, row in enumerate(rows, start=2):
            probs = [float(row[name]) for name in prob_cols]
            require(all(math.isfinite(value) and 0 <= value <= 1 for value in probs),
                    f"invalid probabilities at CSV row {index}")
            require(abs(sum(probs) - 1.0) <= 2e-5,
                    f"probability sum invalid at CSV row {index}: {sum(probs)}")
        result_data = json.loads(results.read_text(encoding="utf-8"))
        require(result_data.get("n_total") == 671, "results.json n_total != 671")

    print(
        json.dumps(
            {
                "status": "PASS",
                "subset": subset,
                "seed": seed,
                "manifest_sha256": registered["sha256"],
                "train_epochs": len(train_rows),
                "evaluation": bool(args.require_eval),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
