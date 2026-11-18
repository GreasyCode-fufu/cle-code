#!/usr/bin/env python3
"""Calibrate new KD components on real training batches without Student updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/cle/sunfuxiang-2/ultralytics-main")
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ultralytics.utils.torch_utils import autocast
from ultralytics_ext.composite_kd import CompositeKDConfig
from ultralytics_ext.composite_kd_trainer import CompositeKDDetectionTrainer
from composite_distill_train import CFG, DATA, STUDENT, TEACHER


def digest(path):
    h = hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()


def args():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--experiment", required=True); p.add_argument("--components", required=True)
    p.add_argument("--allocations", required=True); p.add_argument("--output", required=True); p.add_argument("--teacher", default=str(TEACHER))
    p.add_argument("--student-yaml", default=str(STUDENT)); p.add_argument("--data", default=str(DATA)); p.add_argument("--batches", type=int, default=20)
    p.add_argument("--discriminator-batches", type=int, default=100); p.add_argument("--layer-weights", default="1,1,1"); return p.parse_args()


def next_batch(iterator, loader):
    try: return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader); return next(iterator), iterator


def training_class_weights(dataset, nc=3):
    counts = np.zeros(nc, dtype=np.float64)
    for label in dataset.labels:
        cls = np.asarray(label["cls"], dtype=np.int64).reshape(-1)
        for value in cls:
            if 0 <= value < nc: counts[value] += 1
    raw = 1.0 / np.sqrt(np.maximum(counts, 1.0)); weights = np.clip(raw / raw.mean(), 0.5, 2.0)
    return counts.tolist(), weights.tolist()


def main():
    a = args(); components = [x for x in a.components.split(",") if x]; allocations = json.loads(a.allocations)
    cfg = CompositeKDConfig(experiment=a.experiment, component_weights={x: 1.0 for x in components},
                            class_weights=(1.0, 1.0, 1.0), layer_weights=tuple(float(x) for x in a.layer_weights.split(",")))
    overrides = {"cfg": str(CFG), "model": a.student_yaml, "data": a.data, "epochs": 1, "batch": 32, "imgsz": 640,
                 "workers": 0, "device": "0", "project": str(ROOT / "runs/analysis/nightly_kd_sweep_20260801/calibration"),
                 "name": a.experiment, "seed": 0, "deterministic": True, "amp": True, "optimizer": "auto", "patience": 0,
                 "mosaic": 1.0, "mixup": 0.0, "close_mosaic": 10, "pretrained": True, "resume": False, "val": False,
                 "save": False, "plots": False, "compile": False, "fraction": 1.0, "exist_ok": True}
    trainer = CompositeKDDetectionTrainer(overrides=overrides, composite_config=cfg, teacher_path=a.teacher)
    teacher_file_before = digest(a.teacher); ratios = {x: [] for x in components}; grad_probe = None
    try:
        trainer._setup_train(); trainer._model_train(); counts, class_weights = training_class_weights(trainer.train_loader.dataset)
        if "foreground_cb" in components:
            trainer.composite_loss.foreground_cb.class_weights = torch.tensor(class_weights, device=trainer.device)
            trainer.composite_config.class_weights = tuple(class_weights)
        iterator = iter(trainer.train_loader)
        if trainer.adversarial is not None:
            for i in range(a.discriminator_batches):
                raw, iterator = next_batch(iterator, trainer.train_loader)
                with autocast(trainer.amp): batch = trainer.preprocess_batch(raw)
                step = trainer.compute_kd_step(batch, epoch=20, batch_index=1, number_of_batches=2)
                trainer.clear_step_features(); del step, batch
        for i in range(a.batches):
            raw, iterator = next_batch(iterator, trainer.train_loader)
            with autocast(trainer.amp): batch = trainer.preprocess_batch(raw)
            step = trainer.compute_kd_step(batch, epoch=20, batch_index=0 if i == 0 else 1, number_of_batches=2)
            det = abs(float(step["det_loss"].detach())); bs = int(step["batch_size"])
            for name in components: ratios[name].append(float(step["raw_components"][name].detach()) * bs / max(det, 1e-12))
            if i == 0 and trainer.adversarial is not None: grad_probe = float(step["gan_stats"].get("adv_to_det_grad_ratio", 0.0))
            trainer.clear_step_features(); del step, batch
        medians = {k: statistics.median(v) for k, v in ratios.items()}
        lambda5 = {k: 0.05 / max(v, 1e-12) for k, v in medians.items()}
        default_layers = tuple(float(x) for x in a.layer_weights.split(",")) == (1.0, 1.0, 1.0)
        final_weights = {k: float(allocations[k]) * (0.0431658384 if k == "gram" and default_layers else 0.1024629257 if k == "mtrd" else lambda5[k]) for k in components}
        warning = None; predicted_grad_ratio = None
        adv_names = [x for x in components if x.startswith("adv_")]
        if adv_names and grad_probe is not None:
            # Probe used unit weights at full ramp; apply calibrated weights conservatively by their maximum.
            predicted_grad_ratio = grad_probe * max(final_weights[x] for x in adv_names)
            if predicted_grad_ratio > 0.3:
                factor = 0.3 / predicted_grad_ratio
                for x in adv_names: final_weights[x] *= factor
                warning = f"adversarial weights capped by gradient ratio factor {factor:.6g}"
        result = {"status": "SUCCESS", "experiment": a.experiment, "measurement_batches": a.batches,
                  "discriminator_warmup_batches": a.discriminator_batches if trainer.adversarial else 0,
                  "formula": "ratio=(raw_loss*batch_size)/det_loss; lambda5=0.05/median_ratio",
                  "raw_ratios": ratios, "median_raw_ratios": medians, "lambda_5_percent": lambda5,
                  "allocations": allocations, "final_component_weights": final_weights,
                  "class_counts_training_only": counts, "class_weights": class_weights,
                  "gradient_probe_unit_weight": grad_probe, "predicted_weighted_adv_to_det_grad_ratio": predicted_grad_ratio,
                  "warning": warning, "student_initialization_hash": trainer.student_hash_before_training,
                  "teacher_model_hash": trainer.teacher_hash_before, "teacher_file_hash_before": teacher_file_before,
                  "teacher_file_hash_after": digest(a.teacher), "teacher_eval": not trainer.teacher_model.training}
    finally:
        trainer.clear_step_features(); trainer.close_kd(); torch.cuda.empty_cache()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
