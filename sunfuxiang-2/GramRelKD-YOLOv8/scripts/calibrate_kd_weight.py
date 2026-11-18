#!/usr/bin/env python3
"""Calibrate relation-KD weight from 20 real training batches without backward."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path("/data/cle/sunfuxiang-2/ultralytics-main")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.utils.torch_utils import autocast
from ultralytics_ext import KDConfig, KDDetectionTrainer

TEACHER = ROOT / "runs/detect/runs/detect/hongkong_L1-96/weights/best.pt"
STUDENT = ROOT / "ultralytics/cfg/models/hongkong/yolov8n-L1-96-SR1.yaml"
DATA = ROOT / "ultralytics/cfg/datasets/coco8.yaml"
CFG = ROOT / "ultralytics/cfg/default.yaml"
OUTPUT_DIR = ROOT / "runs/analysis/kd_implementation_check"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default=str(TEACHER))
    parser.add_argument("--student-yaml", default=str(STUDENT))
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--cfg", default=str(CFG))
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=20)
    return parser.parse_args()


def clipped_weight(target: float, median_ratio: float) -> float:
    return min(10.0, max(1e-3, target / median_ratio))


def main() -> int:
    args = parse_args()
    if args.batches <= 0:
        raise ValueError("--batches must be positive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kd_config = KDConfig(
        teacher_path=args.teacher,
        pool_size=args.pool_size,
        kd_weight=1.0,
        warmup_epochs=10.0,
    )
    overrides = {
        "cfg": args.cfg,
        "model": args.student_yaml,
        "data": args.data,
        "epochs": 300,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": 0,
        "device": args.device,
        "project": str(OUTPUT_DIR / "tmp"),
        "name": "calibration",
        "exist_ok": True,
        "seed": args.seed,
        "deterministic": True,
        "amp": True,
        "optimizer": "auto",
        "patience": 0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "close_mosaic": 10,
        "pretrained": True,
        "resume": False,
        "val": False,
        "save": False,
        "plots": False,
        "compile": False,
    }
    trainer = KDDetectionTrainer(overrides=overrides, kd_config=kd_config)
    rows: list[dict[str, float | int]] = []
    try:
        trainer._setup_train()
        trainer._model_train()
        loader = iter(trainer.train_loader)
        number_of_batches = len(trainer.train_loader)
        for index in range(args.batches):
            raw_batch = next(loader)
            with autocast(trainer.amp):
                batch = trainer.preprocess_batch(raw_batch)
            step = trainer.compute_kd_step(
                batch,
                epoch=0,
                batch_index=index,
                number_of_batches=number_of_batches,
            )
            det_loss = float(step["det_loss"].detach())
            kd_raw = float(step["kd_raw"].detach())
            batch_size = int(step["batch_size"])
            raw_scaled = kd_raw * batch_size
            ratio = raw_scaled / det_loss
            rows.append(
                {
                    "batch": index,
                    "batch_size": batch_size,
                    "det_loss": det_loss,
                    "kd_raw": kd_raw,
                    "kd_batch_scaled_raw": raw_scaled,
                    "raw_kd_to_det_ratio": ratio,
                    "p3": float(step["kd_parts"]["p3"].detach()),
                    "p4": float(step["kd_parts"]["p4"].detach()),
                    "p5": float(step["kd_parts"]["p5"].detach()),
                }
            )
            trainer.clear_step_features()
            del raw_batch, batch, step
        det_median = statistics.median(float(row["det_loss"]) for row in rows)
        kd_median = statistics.median(float(row["kd_raw"]) for row in rows)
        ratio_median = statistics.median(float(row["raw_kd_to_det_ratio"]) for row in rows)
        recommendations = {
            "0.025": clipped_weight(0.025, ratio_median),
            "0.05": clipped_weight(0.05, ratio_median),
            "0.10": clipped_weight(0.10, ratio_median),
        }
        result = {
            "status": "SUCCESS",
            "batches": args.batches,
            "batch_size": args.batch,
            "imgsz": args.imgsz,
            "seed": args.seed,
            "student_initialization": "YAML random initialization",
            "backward_executed": False,
            "optimizer_step_executed": False,
            "median_detection_loss": det_median,
            "median_raw_kd_loss": kd_median,
            "median_raw_kd_to_detection_ratio": ratio_median,
            "recommended_kd_weight": recommendations,
            "preferred_target_ratio": 0.05,
            "rows": rows,
            "teacher_optimizer_isolation": trainer.kd_isolation_report,
        }
    finally:
        trainer.clear_step_features()
        trainer.close_kd()
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json_path = OUTPUT_DIR / "kd_weight_calibration.json"
    csv_path = OUTPUT_DIR / "kd_weight_calibration.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"median det loss: {result['median_detection_loss']:.8g}")
    print(f"median raw KD: {result['median_raw_kd_loss']:.8g}")
    print(f"median raw KD/det ratio: {result['median_raw_kd_to_detection_ratio']:.8g}")
    print(f"recommended weight for 2.5%: {recommendations['0.025']:.8g}")
    print(f"recommended weight for 5%: {recommendations['0.05']:.8g}")
    print(f"recommended weight for 10%: {recommendations['0.10']:.8g}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
