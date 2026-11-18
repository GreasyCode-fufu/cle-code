#!/usr/bin/env python3
"""Train or preflight an exploratory composite KD experiment."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/data/cle/sunfuxiang-2/ultralytics-main")
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import autocast, unwrap_model
from ultralytics_ext.composite_kd import CompositeKDConfig
from ultralytics_ext.composite_kd_trainer import CompositeKDDetectionTrainer

TEACHER = ROOT / "runs/detect/runs/detect/hongkong_L1-96/weights/best.pt"
STUDENT = ROOT / "ultralytics/cfg/models/hongkong/yolov8n-L1-96-SR1.yaml"
DATA = ROOT / "ultralytics/cfg/datasets/coco8.yaml"
CFG = ROOT / "ultralytics/cfg/default.yaml"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("inspect", "smoke", "train"), default="inspect")
    p.add_argument("--experiment", required=True); p.add_argument("--weights-json", required=True)
    p.add_argument("--class-weights-json"); p.add_argument("--teacher", default=str(TEACHER)); p.add_argument("--student-yaml", default=str(STUDENT))
    p.add_argument("--data", default=str(DATA)); p.add_argument("--cfg", default=str(CFG)); p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32); p.add_argument("--imgsz", type=int, default=640); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0"); p.add_argument("--seed", type=int, default=0); p.add_argument("--project", default=str(ROOT / "runs/detect"))
    p.add_argument("--name"); p.add_argument("--fraction", type=float, default=1.0); p.add_argument("--layer-weights", default="1,1,1")
    p.add_argument("--result-json")
    return p.parse_args()


def overrides(a):
    result = {"cfg": a.cfg, "model": a.student_yaml, "data": a.data, "epochs": a.epochs, "batch": a.batch, "imgsz": a.imgsz,
              "workers": a.workers, "device": a.device, "project": a.project, "name": a.name or a.experiment, "seed": a.seed,
              "deterministic": True, "amp": True, "optimizer": "auto", "patience": 0, "mosaic": 1.0, "mixup": 0.0,
              "close_mosaic": 10, "pretrained": True, "resume": False, "val": True, "compile": False, "fraction": a.fraction,
              "exist_ok": False}
    if a.mode in {"inspect", "smoke"}:
        result.update({"epochs": 1, "batch": 4, "workers": 0, "fraction": 0.02, "val": False, "plots": False,
                       "project": str(ROOT / "runs/analysis/nightly_kd_sweep_20260801/preflight"),
                       "name": f"{a.experiment}_{a.mode}_{int(time.time())}"})
    return result


def config(a):
    weights = json.loads(Path(a.weights_json).read_text()) if Path(a.weights_json).is_file() else json.loads(a.weights_json)
    class_weights = None
    if a.class_weights_json:
        class_weights = tuple(json.loads(Path(a.class_weights_json).read_text()))
    return CompositeKDConfig(experiment=a.experiment, component_weights={k: float(v) for k, v in weights.items()},
                             class_weights=class_weights, layer_weights=tuple(float(x) for x in a.layer_weights.split(",")))


def checkpoint_clean(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False); model = ckpt.get("model") or ckpt.get("ema")
    keys = model.state_dict() if model is not None else {}
    return not any(any(token in k.lower() for token in ("teacher", "discriminator", "composite")) for k in keys)


def run_inspect(a, kd_config):
    trainer = CompositeKDDetectionTrainer(overrides=overrides(a), composite_config=kd_config, teacher_path=a.teacher)
    temp = ROOT / "runs/analysis/nightly_kd_sweep_20260801/preflight" / f"{a.experiment}_temporary_student.pt"
    try:
        trainer._setup_train(); trainer._model_train(); raw = next(iter(trainer.train_loader))
        with autocast(trainer.amp): batch = trainer.preprocess_batch(raw)
        before = trainer.student_hash_before_training
        step = trainer.compute_kd_step(batch, epoch=0, batch_index=0, number_of_batches=len(trainer.train_loader))
        trainer.scaler.scale(step["total_loss"]).backward(); trainer.optimizer_step(); trainer.clear_step_features()
        changed = before != trainer._module_sha256(unwrap_model(trainer.model))
        # Temporary standalone checkpoint must not pickle training-only hook closures.
        trainer.teacher_hooks.remove(); trainer.student_hooks.remove()
        for hook_name in ("teacher_head_hook", "student_head_hook"):
            hook = getattr(trainer, hook_name, None)
            if hook is not None: hook.remove(); setattr(trainer, hook_name, None)
        temp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": copy.deepcopy(unwrap_model(trainer.model)).half(), "ema": None, "train_args": dict(vars(trainer.args))}, temp)
        loaded = YOLO(str(temp)); standalone = sum(p.numel() for p in loaded.model.parameters()) == sum(p.numel() for p in unwrap_model(trainer.model).parameters())
        return {"status": "SUCCESS", "student_initialization_hash": before, "student_changed": changed,
                "teacher_hash_unchanged": trainer._module_sha256(trainer.teacher_model) == trainer.teacher_hash_before,
                "teacher_eval": not trainer.teacher_model.training, "hook_counts": [step["teacher_hook_counts"], step["student_hook_counts"]],
                "finite": bool(torch.isfinite(step["total_loss"])), "checkpoint_standalone": standalone and checkpoint_clean(temp),
                "det_loss": float(step["det_loss"]), "kd_scaled": float(step["kd_scaled"]), "isolation": trainer.kd_isolation_report}
    finally:
        trainer.clear_step_features(); trainer.close_kd(); temp.unlink(missing_ok=True)


def run_train(a, kd_config):
    holders = []
    def factory(*, overrides, _callbacks=None):
        t = CompositeKDDetectionTrainer(overrides=overrides, _callbacks=_callbacks, composite_config=kd_config, teacher_path=a.teacher); holders.append(t); return t
    started = time.perf_counter(); teacher_before = sha256(a.teacher)
    model = YOLO(a.student_yaml); model.train(trainer=factory, **overrides(a)); elapsed = time.perf_counter() - started
    t = holders[-1]; checkpoint = t.best if t.best.exists() else t.last
    result = {"status": "SUCCESS", "mode": a.mode, "experiment": a.experiment, "elapsed_seconds": elapsed, "save_dir": str(t.save_dir),
              "best": str(t.best), "last": str(t.last), "checkpoint": str(checkpoint), "metrics_csv": str(t.kd_metrics_path),
              "student_initialization_hash": t.student_hash_before_training, "student_changed": t.student_parameters_changed,
              "teacher_file_hash_before": teacher_before, "teacher_file_hash_after": sha256(a.teacher), "teacher_hash_unchanged": t.teacher_hash_unchanged,
              "teacher_eval_always": t.teacher_eval_always, "checkpoint_clean": checkpoint_clean(checkpoint),
              "peak_cuda_memory_mb": t.kd_peak_cuda_memory_mb, "component_weights": kd_config.component_weights, "isolation": t.kd_isolation_report}
    output = Path(t.save_dir) / "composite_training_result.json"; output.write_text(json.dumps(result, indent=2) + "\n")
    del model; holders.clear(); gc.collect(); torch.cuda.empty_cache(); return result


def main():
    a = parse_args(); kd_config = config(a)
    for path in (a.teacher, a.student_yaml, a.data, a.cfg):
        if not Path(path).is_file(): raise FileNotFoundError(path)
    result = run_inspect(a, kd_config) if a.mode == "inspect" else run_train(a, kd_config)
    if a.result_json:
        Path(a.result_json).parent.mkdir(parents=True, exist_ok=True); Path(a.result_json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
