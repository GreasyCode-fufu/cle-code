#!/usr/bin/env python3
"""Compare detection, calibrated MTRD, and Gram-RelKD feature gradients on train batches."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/data/cle/sunfuxiang-2/ultralytics-main")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.utils.torch_utils import autocast, unwrap_model
from ultralytics_ext.markov_kd_trainer import MTRDConfig, MarkovKDDetectionTrainer
from ultralytics_ext.relation_kd import ChannelRelationKDLoss

TEACHER = ROOT / "runs/detect/runs/detect/hongkong_L1-96/weights/best.pt"
STUDENT = ROOT / "ultralytics/cfg/models/hongkong/yolov8n-L1-96-SR1.yaml"
DATA = ROOT / "ultralytics/cfg/datasets/coco8.yaml"
CFG = ROOT / "ultralytics/cfg/default.yaml"
OUTPUT_DIR = ROOT / "runs/analysis/mtrd_implementation_check"
TEMPERATURE_JSON = OUTPUT_DIR / "markov_temperature_calibration.json"
WEIGHT_JSON = OUTPUT_DIR / "markov_kd_weight_calibration.json"
LAYERS = (15, 18, 21)
NAMES = ("p3", "p4", "p5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default=str(TEACHER))
    parser.add_argument("--student-yaml", default=str(STUDENT))
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--cfg", default=str(CFG))
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def tensor_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().float()))


def main() -> int:
    args = parse_args()
    if args.batches <= 0:
        raise ValueError("--batches must be positive")
    temperature_result = json.loads(TEMPERATURE_JSON.read_text(encoding="utf-8"))
    weight_result = json.loads(WEIGHT_JSON.read_text(encoding="utf-8"))
    temperature = float(temperature_result["recommended_temperature"])
    weight = float(weight_result["recommended_kd_weight"]["0.05"])
    config = MTRDConfig(
        teacher_path=args.teacher,
        temperature=temperature,
        transition_steps=(1, 2),
        step_weights=(1.0, 1.0),
        self_loop_prob=0.0,
        kd_weight=weight,
        warmup_epochs=10.0,
        log_dir=str(OUTPUT_DIR / "gradient_diagnostics"),
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
        "name": "gradient_diagnostics",
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
    trainer = MarkovKDDetectionTrainer(overrides=overrides, mtrd_config=config)
    rows: list[dict[str, object]] = []
    result: dict[str, object] = {}
    try:
        trainer._setup_train()
        trainer._model_train()
        gram_loss = ChannelRelationKDLoss(
            layer_indices=LAYERS, layer_weights=(1.0, 1.0, 1.0), pool_size=20
        ).to(trainer.device)
        student_hash_before = trainer._module_sha256(unwrap_model(trainer.model))
        student_state_before = {
            key: value.detach().clone() for key, value in unwrap_model(trainer.model).state_dict().items()
        }
        teacher_hash_before = trainer._module_sha256(trainer.teacher_model)
        loader = iter(trainer.train_loader)
        number_of_batches = len(trainer.train_loader)
        for batch_index in range(args.batches):
            raw_batch = next(loader)
            with autocast(trainer.amp):
                batch = trainer.preprocess_batch(raw_batch)
            step = trainer.compute_kd_step(
                batch, epoch=0, batch_index=batch_index, number_of_batches=number_of_batches
            )
            student_features = trainer.student_hooks.get()
            teacher_features = trainer.teacher_hooks.get()
            gram_raw, _ = gram_loss(student_features, teacher_features)
            weighted_mtrd = step["kd_raw"] * weight * int(step["batch_size"])
            for layer, name in zip(LAYERS, NAMES):
                feature = student_features[layer]
                det_gradient = torch.autograd.grad(
                    step["det_loss"], feature, retain_graph=True, create_graph=False
                )[0]
                mtrd_gradient = torch.autograd.grad(
                    weighted_mtrd, feature, retain_graph=True, create_graph=False
                )[0]
                gram_gradient = torch.autograd.grad(
                    gram_raw, feature, retain_graph=True, create_graph=False
                )[0]
                det_norm = tensor_norm(det_gradient)
                mtrd_norm = tensor_norm(mtrd_gradient)
                gram_norm = tensor_norm(gram_gradient)
                ratio = mtrd_norm / max(det_norm, 1e-12)
                cosine = float(
                    F.cosine_similarity(
                        mtrd_gradient.detach().float().flatten(),
                        gram_gradient.detach().float().flatten(),
                        dim=0,
                        eps=1e-12,
                    )
                )
                rows.append(
                    {
                        "batch": batch_index,
                        "scale": name,
                        "layer": layer,
                        "det_grad_norm": det_norm,
                        "mtrd_grad_norm": mtrd_norm,
                        "weighted_mtrd_to_det_grad_ratio": ratio,
                        "gram_grad_norm": gram_norm,
                        "mtrd_vs_gram_gradient_cosine": cosine,
                    }
                )
            trainer.clear_step_features()
            del raw_batch, batch, step, student_features, teacher_features, gram_raw

        summary: dict[str, object] = {}
        for name in NAMES:
            scale_rows = [row for row in rows if row["scale"] == name]
            summary[name] = {
                key: statistics.median(float(row[key]) for row in scale_rows)
                for key in (
                    "det_grad_norm",
                    "mtrd_grad_norm",
                    "weighted_mtrd_to_det_grad_ratio",
                    "gram_grad_norm",
                    "mtrd_vs_gram_gradient_cosine",
                )
            }
        gradient_warning = any(
            summary[name]["weighted_mtrd_to_det_grad_ratio"] > 0.5 for name in NAMES
        )
        redundancy_risk = all(
            summary[name]["mtrd_vs_gram_gradient_cosine"] > 0.95 for name in NAMES
        )
        unwrap_model(trainer.model).load_state_dict(student_state_before)
        result = {
            "status": "SUCCESS",
            "temperature": temperature,
            "recommended_5_percent_weight": weight,
            "batches": args.batches,
            "batch_size": args.batch,
            "imgsz": args.imgsz,
            "seed": args.seed,
            "data_source": "train split only",
            "optimizer_step_executed": False,
            "backward_used": "torch.autograd.grad only",
            "student_initialization": "YAML random initialization",
            "summary_median": summary,
            "rows": rows,
            "gradient_ratio_status": "WARNING" if gradient_warning else "PASS",
            "gradient_warning_threshold": 0.5,
            "redundancy_status": "HIGH_REDUNDANCY_RISK" if redundancy_risk else "DISTINCT_GRADIENT_SIGNAL",
            "redundancy_threshold": 0.95,
            "diagnostic_scope": "five batches; not evidence of long-term training behavior",
            "student_state_restored": True,
            "student_hash_before": student_hash_before,
            "student_hash_after": trainer._module_sha256(unwrap_model(trainer.model)),
            "teacher_hash_before": teacher_hash_before,
            "teacher_hash_after": trainer._module_sha256(trainer.teacher_model),
        }
        result["student_parameters_unchanged"] = result["student_hash_before"] == result["student_hash_after"]
        result["teacher_hash_unchanged"] = result["teacher_hash_before"] == result["teacher_hash_after"]
    finally:
        trainer.clear_step_features()
        trainer.close_kd()
        result["hooks_removed"] = trainer.kd_hooks_removed
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json_path = OUTPUT_DIR / "markov_gradient_diagnostics.json"
    csv_path = OUTPUT_DIR / "markov_gradient_diagnostics.csv"
    report_path = OUTPUT_DIR / "markov_vs_gram_report.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path.write_text(
        "# MTRD vs Gram RelKD Gradient Diagnostic\n\n"
        f"- Gradient ratio status: {result['gradient_ratio_status']}\n"
        f"- Redundancy status: {result['redundancy_status']}\n"
        "- Scope: five real training batches; no optimizer step.\n\n"
        "| Scale | Det grad | Weighted MTRD grad | MTRD/Det | Gram grad | MTRD-vs-Gram cosine |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {name.upper()} | {result['summary_median'][name]['det_grad_norm']:.6g} | "
            f"{result['summary_median'][name]['mtrd_grad_norm']:.6g} | "
            f"{result['summary_median'][name]['weighted_mtrd_to_det_grad_ratio']:.6g} | "
            f"{result['summary_median'][name]['gram_grad_norm']:.6g} | "
            f"{result['summary_median'][name]['mtrd_vs_gram_gradient_cosine']:.6g} |"
            for name in NAMES
        )
        + "\n",
        encoding="utf-8",
    )
    for name in NAMES:
        print(
            f"{name.upper()} weighted MTRD/detection gradient ratio: "
            f"{result['summary_median'][name]['weighted_mtrd_to_det_grad_ratio']:.8g}"
        )
        print(
            f"{name.upper()} MTRD-vs-Gram cosine: "
            f"{result['summary_median'][name]['mtrd_vs_gram_gradient_cosine']:.8g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
