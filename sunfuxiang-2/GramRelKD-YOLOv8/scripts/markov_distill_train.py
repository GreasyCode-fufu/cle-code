#!/usr/bin/env python3
"""Independent MTRD entry point that preserves the verified Gram-RelKD defaults."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path("/data/cle/sunfuxiang-2/ultralytics-main")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import autocast, unwrap_model
from ultralytics_ext.markov_kd_trainer import MTRDConfig, MarkovKDDetectionTrainer

DEFAULT_TEACHER = ROOT / "runs/detect/runs/detect/hongkong_L1-96/weights/best.pt"
DEFAULT_STUDENT = ROOT / "ultralytics/cfg/models/hongkong/yolov8n-L1-96-SR1.yaml"
DEFAULT_DATA = ROOT / "ultralytics/cfg/datasets/coco8.yaml"
DEFAULT_CFG = ROOT / "ultralytics/cfg/default.yaml"
RELKD_ARGS = ROOT / "runs/detect/hongkong_SR1_P4Only_RelKD/args.yaml"
ANALYSIS_DIR = ROOT / "runs/analysis/mtrd_implementation_check"


def load_formal_defaults() -> dict[str, Any]:
    if not RELKD_ARGS.is_file():
        raise FileNotFoundError(f"successful RelKD args.yaml not found: {RELKD_ARGS}")
    values = yaml.safe_load(RELKD_ARGS.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError(f"RelKD args must be a mapping, got {type(values).__name__}")
    return values


def parse_csv_ints(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("value must not be empty")
    return values


def parse_csv_floats(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not values:
        raise argparse.ArgumentTypeError("value must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    formal = load_formal_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "smoke", "train"), default="inspect")
    parser.add_argument("--teacher", default=str(DEFAULT_TEACHER))
    parser.add_argument("--student-yaml", default=str(formal.get("model", DEFAULT_STUDENT)))
    parser.add_argument("--data", default=str(formal.get("data", DEFAULT_DATA)))
    parser.add_argument("--cfg", default=str(formal.get("cfg", DEFAULT_CFG)))
    parser.add_argument("--epochs", type=int, default=int(formal["epochs"]))
    parser.add_argument("--batch", type=int, default=int(formal["batch"]))
    parser.add_argument("--imgsz", type=int, default=int(formal["imgsz"]))
    parser.add_argument("--workers", type=int, default=int(formal["workers"]))
    parser.add_argument("--device", default=str(formal["device"]))
    parser.add_argument("--seed", type=int, default=int(formal["seed"]))
    parser.add_argument("--name", default="hongkong_SR1_P4Only_MTRD")
    parser.add_argument("--project", default=str(formal["project"]))
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--kd-warmup-epochs", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--transition-steps", type=parse_csv_ints, default=(1, 2))
    parser.add_argument("--step-weights", type=parse_csv_floats, default=(1.0, 1.0))
    parser.add_argument("--self-loop-prob", type=float, default=0.0)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--layer-weights", type=parse_csv_floats, default=(1.0, 1.0, 1.0))
    return parser.parse_args()


def make_mtrd_config(args: argparse.Namespace, *, log_dir: str | None = None) -> MTRDConfig:
    return MTRDConfig(
        teacher_path=args.teacher,
        layer_indices=(15, 18, 21),
        layer_weights=args.layer_weights,
        pool_size=args.pool_size,
        temperature=args.temperature,
        transition_steps=args.transition_steps,
        step_weights=args.step_weights,
        self_loop_prob=args.self_loop_prob,
        divergence="kl",
        kd_weight=args.kd_weight,
        warmup_epochs=args.kd_warmup_epochs,
        eps=1e-6,
        center=True,
        log_dir=log_dir,
    )


def baseline_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Read every default from the successful RelKD args, then apply explicit CLI identity fields."""
    overrides = load_formal_defaults().copy()
    for key in ("save_dir",):
        overrides.pop(key, None)
    overrides.update(
        {
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
            "resume": False,
            "compile": False,
        }
    )
    return overrides


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
            "name": "mtrd_inspect",
            "exist_ok": True,
        }
    )
    trainer = MarkovKDDetectionTrainer(overrides=overrides, mtrd_config=make_mtrd_config(args))
    try:
        trainer._setup_train()
        trainer._model_train()
        student_state_before = {
            key: value.detach().clone() for key, value in unwrap_model(trainer.model).state_dict().items()
        }
        raw_batch = next(iter(trainer.train_loader))
        with autocast(trainer.amp):
            batch = trainer.preprocess_batch(raw_batch)
        step = trainer.compute_kd_step(
            batch, epoch=0, batch_index=0, number_of_batches=len(trainer.train_loader)
        )
        result = {
            "status": "SUCCESS",
            "mode": "inspect",
            "formal_defaults_source": str(RELKD_ARGS),
            "student_initialization": "YAML random initialization",
            "teacher_shapes": {
                str(index): list(value.shape) for index, value in trainer.teacher_hooks.get().items()
            },
            "student_shapes": {
                str(index): list(value.shape) for index, value in trainer.student_hooks.get().items()
            },
            "det_loss": float(step["det_loss"].detach()),
            "mtrd_raw": float(step["kd_raw"].detach()),
            "mtrd_scaled": float(step["kd_scaled"].detach()),
            "mtrd_parts": {key: float(value.detach()) for key, value in step["kd_parts"].items()},
            "kd_ramp": step["kd_ramp"],
            "teacher_hook_counts": step["teacher_hook_counts"],
            "student_hook_counts": step["student_hook_counts"],
            "isolation": trainer.kd_isolation_report,
        }
        unwrap_model(trainer.model).load_state_dict(student_state_before)
        result["student_state_restored"] = True
        return result
    finally:
        trainer.clear_step_features()
        trainer.close_kd()


def checkpoint_absent_teacher(path: Path) -> bool:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("model") or checkpoint.get("ema")
    return model is not None and not any("teacher" in key.lower() for key in model.state_dict())


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
                "project": str(ANALYSIS_DIR / "tmp"),
                "name": "hongkong_SR1_P4Only_MTRD_smoke",
                "exist_ok": True,
            }
        )

    config = make_mtrd_config(args)
    holder: list[MarkovKDDetectionTrainer] = []

    def trainer_factory(*, overrides: dict[str, Any], _callbacks=None):
        trainer = MarkovKDDetectionTrainer(
            overrides=overrides, _callbacks=_callbacks, mtrd_config=config
        )
        if is_smoke:
            # Ultralytics 8.4.84 validates on final_epoch even when args.val=False.
            # The requested smoke contract explicitly forbids validation, so suppress
            # only these instance calls; mode=train keeps the verified base behavior.
            trainer.validate = lambda: ({}, 0.0)
            trainer.final_eval = lambda: None
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
    records_finite = all(
        all(
            isinstance(value, (int, float)) and torch.isfinite(torch.tensor(float(value)))
            for value in row.values()
        )
        for row in trainer.kd_epoch_records
    )
    memory_trace = trainer.mtrd_batch_memory_allocated_mb
    memory_tail = memory_trace[3:] if len(memory_trace) > 3 else memory_trace
    sustained_memory_growth = len(memory_tail) >= 4 and all(
        right > left + 1.0 for left, right in zip(memory_tail, memory_tail[1:])
    )
    result = {
        "status": "SUCCESS",
        "mode": args.mode,
        "validation_executed": not is_smoke,
        "elapsed_seconds": elapsed,
        "save_dir": str(trainer.save_dir),
        "mtrd_metrics_csv": str(trainer.kd_metrics_path),
        "mtrd_metrics_exists": trainer.kd_metrics_path.is_file(),
        "mtrd_epoch_records": trainer.kd_epoch_records,
        "mtrd_metrics_finite": records_finite,
        "batch_memory_allocated_mb": memory_trace,
        "sustained_batch_memory_growth": sustained_memory_growth,
        "teacher_hash_unchanged": trainer.teacher_hash_unchanged,
        "teacher_eval_always": trainer.teacher_eval_always,
        "teacher_absent_from_optimizer": trainer.kd_isolation_report.get("teacher_absent_from_optimizer"),
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
        device = torch.device("cuda:0")
        result["cuda_memory_allocated_after_cleanup_mb"] = torch.cuda.memory_allocated(device) / 1024**2
        result["cuda_memory_reserved_after_cleanup_mb"] = torch.cuda.memory_reserved(device) / 1024**2
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
