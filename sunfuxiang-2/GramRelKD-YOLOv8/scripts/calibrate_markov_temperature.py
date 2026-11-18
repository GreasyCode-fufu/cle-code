#!/usr/bin/env python3
"""Select MTRD temperature from Teacher transition entropy on real training batches only."""

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
from ultralytics_ext.markov_kd_trainer import MTRDConfig, MarkovKDDetectionTrainer
from ultralytics_ext.markov_relation_kd import MultiStepTransitionKDLoss

TEACHER = ROOT / "runs/detect/runs/detect/hongkong_L1-96/weights/best.pt"
STUDENT = ROOT / "ultralytics/cfg/models/hongkong/yolov8n-L1-96-SR1.yaml"
DATA = ROOT / "ultralytics/cfg/datasets/coco8.yaml"
CFG = ROOT / "ultralytics/cfg/default.yaml"
OUTPUT_DIR = ROOT / "runs/analysis/mtrd_implementation_check"
TEMPERATURES = (0.05, 0.10, 0.20, 0.50, 1.00)
LAYERS = (15, 18, 21)
NAMES = ("p3", "p4", "p5")


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


def main() -> int:
    args = parse_args()
    if args.batches <= 0:
        raise ValueError("--batches must be positive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = MTRDConfig(
        teacher_path=args.teacher,
        pool_size=args.pool_size,
        temperature=0.2,
        kd_weight=1.0,
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
        "name": "temperature_calibration",
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
    collected = {
        temperature: {
            name: {key: [] for key in ("entropy", "max_prob", "top5_mass", "step2_entropy")}
            for name in NAMES
        }
        for temperature in TEMPERATURES
    }
    result: dict[str, object] = {}
    try:
        trainer._setup_train()
        trainer._model_train()
        loader = iter(trainer.train_loader)
        for batch_index in range(args.batches):
            raw_batch = next(loader)
            with autocast(trainer.amp):
                batch = trainer.preprocess_batch(raw_batch)
            trainer.clear_step_features()
            trainer.teacher_hooks.enable()
            trainer.teacher_model.eval()
            with autocast(trainer.amp), torch.no_grad():
                trainer.teacher_model(batch["img"])
            trainer.teacher_hooks.validate_once()
            teacher_features = trainer.teacher_hooks.get()
            for temperature in TEMPERATURES:
                objective = MultiStepTransitionKDLoss(
                    pool_size=args.pool_size,
                    temperature=temperature,
                    transition_steps=(1, 2),
                    step_weights=(1.0, 1.0),
                    self_loop_prob=0.0,
                ).to(trainer.device)
                for layer, name in zip(LAYERS, NAMES):
                    diagnostics = objective.transition_diagnostics(teacher_features[layer], teacher=True)
                    mapping = {
                        "entropy": "normalized_entropy_rows",
                        "max_prob": "max_probability_rows",
                        "top5_mass": "top5_mass_rows",
                        "step2_entropy": "step2_normalized_entropy_rows",
                    }
                    for output_name, tensor_name in mapping.items():
                        collected[temperature][name][output_name].extend(
                            diagnostics[tensor_name].float().cpu().flatten().tolist()
                        )
            trainer.clear_step_features()
            del raw_batch, batch, teacher_features

        rows: list[dict[str, object]] = []
        by_temperature: dict[str, object] = {}
        candidates: list[tuple[float, float]] = []
        for temperature in TEMPERATURES:
            scale_results: dict[str, object] = {}
            entropies = []
            all_in_range = True
            for name in NAMES:
                values = collected[temperature][name]
                entropy_median = statistics.median(values["entropy"])
                entropy_mean = statistics.fmean(values["entropy"])
                max_probability = statistics.fmean(values["max_prob"])
                top5_mass = statistics.fmean(values["top5_mass"])
                step2_entropy = statistics.fmean(values["step2_entropy"])
                finite = all(
                    torch.isfinite(torch.tensor(number))
                    for group in values.values()
                    for number in group
                )
                in_range = 0.20 <= entropy_median <= 0.80
                all_in_range = all_in_range and in_range and finite
                entropies.append(entropy_median)
                scale_results[name] = {
                    "normalized_row_entropy_median": entropy_median,
                    "normalized_row_entropy_mean": entropy_mean,
                    "max_transition_probability_mean": max_probability,
                    "top5_probability_mass_mean": top5_mass,
                    "step2_normalized_entropy_mean": step2_entropy,
                    "finite": finite,
                    "entropy_in_required_range": in_range,
                }
                rows.append(
                    {
                        "temperature": temperature,
                        "scale": name,
                        **scale_results[name],
                    }
                )
            overall_median = statistics.median(entropies)
            distance = abs(overall_median - 0.50)
            by_temperature[str(temperature)] = {
                "scales": scale_results,
                "overall_scale_entropy_median": overall_median,
                "distance_to_0.50": distance,
                "all_scales_in_required_range": all_in_range,
            }
            if all_in_range:
                candidates.append((distance, temperature))

        if candidates:
            _, recommended = min(candidates)
            selection_status = "PASS"
            selection_reason = "all scale medians are in [0.20,0.80]; closest overall median to 0.50"
        else:
            _, recommended = min(
                (entry["distance_to_0.50"], float(temperature))
                for temperature, entry in by_temperature.items()
            )
            selection_status = "WARNING"
            selection_reason = "no candidate met all entropy bounds; selected closest overall median to 0.50"

        result = {
            "status": "SUCCESS",
            "selection_status": selection_status,
            "selection_reason": selection_reason,
            "recommended_temperature": recommended,
            "candidate_temperatures": list(TEMPERATURES),
            "batches": args.batches,
            "batch_size": args.batch,
            "imgsz": args.imgsz,
            "seed": args.seed,
            "data_source": "train split only",
            "validation_accessed": False,
            "test_accessed": False,
            "backward_executed": False,
            "optimizer_step_executed": False,
            "student_initialization": "YAML random initialization",
            "results": by_temperature,
            "teacher_optimizer_isolation": trainer.kd_isolation_report,
        }
    finally:
        trainer.clear_step_features()
        trainer.close_kd()
        result["teacher_hash_unchanged"] = trainer.teacher_hash_unchanged
        result["student_parameters_unchanged"] = not trainer.student_parameters_changed
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json_path = OUTPUT_DIR / "markov_temperature_calibration.json"
    csv_path = OUTPUT_DIR / "markov_temperature_calibration.csv"
    report_path = OUTPUT_DIR / "markov_temperature_report.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = result["results"][str(result["recommended_temperature"])]["scales"]
    report_path.write_text(
        "# MTRD Temperature Calibration\n\n"
        f"- Recommended temperature: {result['recommended_temperature']}\n"
        f"- Selection status: {result['selection_status']}\n"
        f"- Selection reason: {result['selection_reason']}\n"
        "- Source: 20 real training batches; no validation/test metrics, backward, or optimizer step.\n\n"
        "| Scale | Entropy median | Entropy mean | Max probability | Top-5 mass | Step-2 entropy |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {name.upper()} | {selected[name]['normalized_row_entropy_median']:.6f} | "
            f"{selected[name]['normalized_row_entropy_mean']:.6f} | "
            f"{selected[name]['max_transition_probability_mean']:.6f} | "
            f"{selected[name]['top5_probability_mass_mean']:.6f} | "
            f"{selected[name]['step2_normalized_entropy_mean']:.6f} |"
            for name in NAMES
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"recommended temperature: {result['recommended_temperature']}")
    for name in NAMES:
        print(f"{name.upper()} entropy: {selected[name]['normalized_row_entropy_median']:.8g}")
        print(f"{name.upper()} max probability: {selected[name]['max_transition_probability_mean']:.8g}")
    print(f"selection status: {result['selection_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
