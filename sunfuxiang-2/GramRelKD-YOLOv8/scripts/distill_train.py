#!/usr/bin/env python3
"""Independent entry point for SR1-P4Only relation knowledge distillation."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/data/cle/sunfuxiang-2/ultralytics-main")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import autocast
from ultralytics_ext import KDConfig, KDDetectionTrainer

DEFAULT_TEACHER = ROOT / "runs/detect/runs/detect/hongkong_L1-96/weights/best.pt"
DEFAULT_STUDENT = ROOT / "ultralytics/cfg/models/hongkong/yolov8n-L1-96-SR1.yaml"
DEFAULT_DATA = ROOT / "ultralytics/cfg/datasets/coco8.yaml"
DEFAULT_CFG = ROOT / "ultralytics/cfg/default.yaml"
ANALYSIS_DIR = ROOT / "runs/analysis/kd_implementation_check"


def parse_weights(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("kd-layer-weights must contain comma-separated numbers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default=str(DEFAULT_TEACHER))
    parser.add_argument("--student-yaml", default=str(DEFAULT_STUDENT))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--cfg", default=str(DEFAULT_CFG))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(ROOT / "runs/detect"))
    parser.add_argument("--name", default="hongkong_SR1_P4Only_RelKD")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kd-weight", type=float, required=False, default=1.0)
    parser.add_argument("--kd-warmup-epochs", type=float, default=10.0)
    parser.add_argument("--kd-pool-size", type=int, default=20)
    parser.add_argument("--kd-layer-weights", type=parse_weights, default=(1.0, 1.0, 1.0))
    parser.add_argument("--mode", choices=("inspect", "smoke", "train"), default="inspect")
    return parser.parse_args()


def make_kd_config(args: argparse.Namespace) -> KDConfig:
    return KDConfig(
        teacher_path=args.teacher,
        layer_indices=(15, 18, 21),
        layer_weights=args.kd_layer_weights,
        pool_size=args.kd_pool_size,
        kd_weight=args.kd_weight,
        warmup_epochs=args.kd_warmup_epochs,
        eps=1e-6,
        center=True,
    )


def baseline_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cfg": args.cfg,
        "model": args.student_yaml,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "device": args.device,
        "project": args.project,
        "name": args.name,
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
        "val": True,
        "compile": False,
    }


def run_inspect(args: argparse.Namespace) -> dict[str, Any]:
    overrides = baseline_overrides(args)
    overrides.update(
        {
            "epochs": 1,
            "workers": 0,
            "fraction": 0.02,
            "val": False,
            "save": False,
            "plots": False,
            "project": str(ANALYSIS_DIR / "tmp"),
            "name": "inspect",
            "exist_ok": True,
        }
    )
    trainer = KDDetectionTrainer(overrides=overrides, kd_config=make_kd_config(args))
    try:
        trainer._setup_train()
        trainer._model_train()
        raw_batch = next(iter(trainer.train_loader))
        with autocast(trainer.amp):
            batch = trainer.preprocess_batch(raw_batch)
        step = trainer.compute_kd_step(
            batch,
            epoch=0,
            batch_index=0,
            number_of_batches=len(trainer.train_loader),
        )
        teacher_shapes = {
            str(i): list(tensor.shape) for i, tensor in trainer.teacher_hooks.get().items()
        }
        student_shapes = {
            str(i): list(tensor.shape) for i, tensor in trainer.student_hooks.get().items()
        }
        return {
            "status": "SUCCESS",
            "mode": "inspect",
            "teacher_shapes": teacher_shapes,
            "student_shapes": student_shapes,
            "det_loss": float(step["det_loss"].detach()),
            "kd_raw": float(step["kd_raw"].detach()),
            "kd_scaled": float(step["kd_scaled"].detach()),
            "kd_ramp": step["kd_ramp"],
            "teacher_hook_counts": step["teacher_hook_counts"],
            "student_hook_counts": step["student_hook_counts"],
            "isolation": trainer.kd_isolation_report,
        }
    finally:
        trainer.clear_step_features()
        trainer.close_kd()


def checkpoint_absent_teacher(path: Path) -> bool:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("model") or checkpoint.get("ema")
    if model is None:
        return False
    return not any("teacher" in key.lower() for key in model.state_dict())


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    is_smoke = args.mode == "smoke"
    overrides = baseline_overrides(args)
    if is_smoke:
        overrides.update(
            {
                "epochs": 1,
                "fraction": 0.02,
                "batch": 4,
                "workers": 0,
                "device": "0",
                "imgsz": 640,
                "val": False,
                "save": False,
                "plots": False,
                "seed": 0,
                "name": "hongkong_SR1_P4Only_RelKD_smoke",
                "exist_ok": True,
            }
        )

    kd_config = make_kd_config(args)
    holder: list[KDDetectionTrainer] = []

    def trainer_factory(*, overrides: dict[str, Any], _callbacks=None):
        trainer = KDDetectionTrainer(
            overrides=overrides,
            _callbacks=_callbacks,
            kd_config=kd_config,
        )
        holder.append(trainer)
        return trainer

    started = time.perf_counter()
    model = YOLO(args.student_yaml)
    model.train(trainer=trainer_factory, **overrides)
    elapsed = time.perf_counter() - started
    if not holder:
        raise RuntimeError("trainer factory was not invoked")
    trainer = holder[-1]
    checkpoint_path = trainer.best if trainer.best.exists() else trainer.last
    result = {
        "status": "SUCCESS",
        "mode": args.mode,
        "elapsed_seconds": elapsed,
        "save_dir": str(trainer.save_dir),
        "kd_metrics_csv": str(trainer.kd_metrics_path),
        "kd_metrics_exists": trainer.kd_metrics_path.is_file(),
        "kd_epoch_records": trainer.kd_epoch_records,
        "teacher_hash_unchanged": trainer.teacher_hash_unchanged,
        "teacher_eval_always": trainer.teacher_eval_always,
        "teacher_absent_from_optimizer": trainer.kd_isolation_report.get(
            "teacher_absent_from_optimizer"
        ),
        "teacher_absent_from_ema": trainer.kd_isolation_report.get("teacher_absent_from_ema"),
        "teacher_absent_from_student_checkpoint": checkpoint_absent_teacher(checkpoint_path),
        "student_parameters_changed": trainer.student_parameters_changed,
        "ema_updates": int(trainer.ema.updates) if trainer.ema else 0,
        "hooks_removed": trainer.kd_hooks_removed,
        "peak_cuda_memory_mb": trainer.kd_peak_cuda_memory_mb,
        "checkpoint_path": str(checkpoint_path),
        "version_lock_ok": trainer.kd_version_lock_ok,
        "isolation": trainer.kd_isolation_report,
    }
    del model, trainer
    holder.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        cuda_device = torch.device("cuda:0")
        result["cuda_memory_allocated_after_cleanup_mb"] = (
            torch.cuda.memory_allocated(cuda_device) / 1024**2
        )
        result["cuda_memory_reserved_after_cleanup_mb"] = (
            torch.cuda.memory_reserved(cuda_device) / 1024**2
        )
    return result


def main() -> int:
    args = parse_args()
    for path in (args.teacher, args.student_yaml, args.data, args.cfg):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "inspect":
        result = run_inspect(args)
        output = ANALYSIS_DIR / "inspect_results.json"
    else:
        result = run_training(args)
        output = ANALYSIS_DIR / (
            "smoke_test_results.json" if args.mode == "smoke" else "train_launch_results.json"
        )
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Result path: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
