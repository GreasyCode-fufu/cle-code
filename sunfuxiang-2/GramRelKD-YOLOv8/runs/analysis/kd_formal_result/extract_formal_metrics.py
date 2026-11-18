#!/usr/bin/env python3
"""Extract exact test metrics without changing the original test.py settings."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

from ultralytics import YOLO, __version__
from ultralytics.utils.torch_utils import get_flops


ROOT = Path("/data/cle/sunfuxiang-2/ultralytics-main")
BEST = ROOT / "runs/detect/hongkong_SR1_P4Only_RelKD/weights/best.pt"
OUT = ROOT / "runs/analysis/kd_formal_result/formal_test_metrics_raw.json"


def scalar(value) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def main() -> None:
    extension_loaded_before = any(
        name == "ultralytics_ext" or name.startswith("ultralytics_ext.") for name in sys.modules
    )
    checkpoint = torch.load(BEST, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint.get("model") or checkpoint.get("ema")
    if checkpoint_model is None:
        raise RuntimeError("best.pt does not contain a model")
    checkpoint_keys = list(checkpoint_model.state_dict())
    teacher_keys = [key for key in checkpoint_keys if "teacher" in key.lower()]
    unfused_params = sum(parameter.numel() for parameter in checkpoint_model.parameters())
    model_yaml = str(getattr(checkpoint_model, "yaml_file", ""))
    model_name = str(getattr(checkpoint_model, "model_name", ""))

    model = YOLO(str(BEST))
    metrics = model.val(**{"cfg": "ultralytics/cfg/default.yaml"})
    box = metrics.box
    all_ap = box.all_ap
    class_indices = [int(index) for index in box.ap_class_index.tolist()]
    class_rows = {}
    for row, class_index in enumerate(class_indices):
        class_name = metrics.names[class_index]
        class_rows[class_name] = {
            "class_index": class_index,
            "precision": scalar(box.p[row]),
            "recall": scalar(box.r[row]),
            "ap50": scalar(box.ap50[row]),
            "ap75": scalar(all_ap[row, 5]),
            "ap50_95": scalar(box.ap[row]),
        }

    fused_model = copy.deepcopy(checkpoint_model).float().fuse(verbose=False)
    fused_params = sum(parameter.numel() for parameter in fused_model.parameters())
    gflops = float(get_flops(fused_model, imgsz=640))
    extension_loaded_after = any(
        name == "ultralytics_ext" or name.startswith("ultralytics_ext.") for name in sys.modules
    )
    output = {
        "status": "SUCCESS",
        "ultralytics_version": __version__,
        "weight_path": str(BEST),
        "weight_size_bytes": BEST.stat().st_size,
        "weight_size_mib": BEST.stat().st_size / (1024**2),
        "evaluation_call": "model.val(**{'cfg':'ultralytics/cfg/default.yaml'})",
        "split": "test",
        "save_dir": str(metrics.save_dir),
        "overall": {
            "precision": scalar(box.mp),
            "recall": scalar(box.mr),
            "map50": scalar(box.map50),
            "map75": scalar(all_ap[:, 5].mean()),
            "map50_95": scalar(box.map),
        },
        "classes": class_rows,
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "unfused_params": unfused_params,
        "fused_params": fused_params,
        "gflops": gflops,
        "model_yaml_file": model_yaml,
        "model_name": model_name,
        "teacher_related_checkpoint_keys": teacher_keys,
        "teacher_absent_from_checkpoint": not teacher_keys,
        "ultralytics_ext_loaded_before_model": extension_loaded_before,
        "ultralytics_ext_loaded_after_evaluation": extension_loaded_after,
        "inference_independent_of_ultralytics_ext": not extension_loaded_before
        and not extension_loaded_after,
    }
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
