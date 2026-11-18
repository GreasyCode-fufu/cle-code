#!/usr/bin/env python3
"""One-batch, no-backward KD pipeline inspection for the local Ultralytics checkout."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/data/cle/sunfuxiang-2/ultralytics-main")
TEACHER_PT = ROOT / "runs/detect/runs/detect/hongkong_L1-96/weights/best.pt"
STUDENT_YAML = ROOT / "ultralytics/cfg/models/hongkong/yolov8n-L1-96-SR1.yaml"
STUDENT_PT = ROOT / "runs/detect/train-9/weights/best.pt"
DATA_YAML = ROOT / "ultralytics/cfg/datasets/coco8.yaml"
OUTPUT_DIR = ROOT / "runs/analysis/kd_code_inspection"
OUTPUT_JSON = OUTPUT_DIR / "kd_feature_shapes.json"
LAYERS = (15, 18, 21)


def jsonable(value: Any) -> Any:
    """Convert common diagnostic values to JSON-compatible objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() <= 20:
            return value.detach().cpu().tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def parameter_digest(model: torch.nn.Module) -> str:
    """Hash parameter values only, excluding mutable BatchNorm buffers."""
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_summary(path: Path) -> dict[str, Any]:
    """Inspect an explicitly trusted local Ultralytics checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("model") or checkpoint.get("ema")
    yaml_data = getattr(model, "yaml", None)
    return {
        "path": str(path),
        "checkpoint_type": type(checkpoint).__name__,
        "checkpoint_keys": sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
        "model_source_key": "model" if checkpoint.get("model") is not None else "ema",
        "model_class": type(model).__name__ if model is not None else None,
        "yaml_saved": isinstance(yaml_data, dict),
        "yaml": jsonable(yaml_data),
        "train_args": jsonable(checkpoint.get("train_args", {})),
        "epoch": checkpoint.get("epoch"),
        "version": checkpoint.get("version"),
        "git": jsonable(checkpoint.get("git")),
    }


def model_container(model: torch.nn.Module) -> tuple[torch.nn.Sequential, str]:
    """Resolve the actual layer container used by the loaded DetectionModel."""
    container = getattr(model, "model", None)
    if not isinstance(container, torch.nn.Sequential):
        raise TypeError(f"Expected model.model to be Sequential, got {type(container).__name__}")
    return container, f"{type(model).__name__}.model"


def layer_summary(container: torch.nn.Sequential, index: int) -> dict[str, Any]:
    """Summarize layer type, repeat count, hidden width, and output width."""
    module = container[index]
    output_channels = None
    cv2 = getattr(module, "cv2", None)
    conv = getattr(cv2, "conv", None)
    if conv is not None:
        output_channels = getattr(conv, "out_channels", None)
    return {
        "index": index,
        "type": type(module).__name__,
        "hidden": getattr(module, "c", None),
        "output_channels": output_channels,
        "internal_repeats": len(module.m) if hasattr(module, "m") else None,
        "parameter_count": sum(p.numel() for p in module.parameters()),
    }


def tensor_output(output: Any) -> torch.Tensor:
    """Require a tensor hook output for neck feature layers."""
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected Tensor hook output, got {type(output).__name__}")
    return output


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import autocast, get_flops

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda:0")
    for required in (TEACHER_PT, STUDENT_YAML, STUDENT_PT, DATA_YAML):
        if not required.is_file():
            raise FileNotFoundError(required)

    result: dict[str, Any] = {
        "status": "RUNNING",
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "device": str(device),
        },
        "paths": {
            "teacher": str(TEACHER_PT),
            "student_yaml": str(STUDENT_YAML),
            "student_best": str(STUDENT_PT),
            "data_yaml": str(DATA_YAML),
        },
        "checkpoint": {
            "teacher": checkpoint_summary(TEACHER_PT),
            "student": checkpoint_summary(STUDENT_PT),
        },
    }

    teacher = YOLO(str(TEACHER_PT)).model.float().to(device)
    student = YOLO(str(STUDENT_PT)).model.float().to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.train()
    for parameter in student.parameters():
        if parameter.is_floating_point():
            parameter.requires_grad_(True)

    teacher_container, teacher_container_path = model_container(teacher)
    student_container, student_container_path = model_container(student)
    result["model"] = {
        "teacher": {
            "class": type(teacher).__name__,
            "container_path": teacher_container_path,
            "parameter_count": sum(p.numel() for p in teacher.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in teacher.parameters() if p.requires_grad),
            "gflops_640": float(get_flops(teacher, imgsz=640)),
            "yaml_saved_on_loaded_model": isinstance(getattr(teacher, "yaml", None), dict),
            "layers_12_18_21": [layer_summary(teacher_container, i) for i in (12, 18, 21)],
        },
        "student": {
            "class": type(student).__name__,
            "container_path": student_container_path,
            "parameter_count": sum(p.numel() for p in student.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in student.parameters() if p.requires_grad),
            "gflops_640": float(get_flops(student, imgsz=640)),
            "yaml_saved_on_loaded_model": isinstance(getattr(student, "yaml", None), dict),
            "layers_12_18_21": [layer_summary(student_container, i) for i in (12, 18, 21)],
        },
    }

    overrides = {
        "model": str(STUDENT_PT),
        "data": str(DATA_YAML),
        "imgsz": 640,
        "batch": 32,
        "device": 0,
        "workers": 0,
        "cache": False,
        "plots": False,
        "amp": True,
        "mosaic": 1.0,
        "mixup": 0.0,
        "close_mosaic": 10,
        "project": "/tmp/codex_kd_inspection",
        "name": "one_batch",
        "exist_ok": True,
        "mode": "train",
    }
    trainer = DetectionTrainer(overrides=overrides)
    trainer.model = student
    trainer.device = device
    trainer.stride = max(int(student.stride.max()), 32)
    trainer.set_model_attributes()
    train_loader = trainer.get_dataloader(trainer.data["train"], batch_size=32, rank=-1, mode="train")
    raw_batch = next(iter(train_loader))
    batch = trainer.preprocess_batch(raw_batch)

    batch_info: dict[str, Any] = {
        "keys": list(batch.keys()),
        "img_shape": list(batch["img"].shape),
        "img_dtype": str(batch["img"].dtype),
        "img_device": str(batch["img"].device),
        "img_min": float(batch["img"].min()),
        "img_max": float(batch["img"].max()),
        "batch_idx_shape": list(batch["batch_idx"].shape),
        "cls_shape": list(batch["cls"].shape),
        "bboxes_shape": list(batch["bboxes"].shape),
        "bboxes_min": float(batch["bboxes"].min()) if batch["bboxes"].numel() else None,
        "bboxes_max": float(batch["bboxes"].max()) if batch["bboxes"].numel() else None,
        "bboxes_first_five": batch["bboxes"][:5].detach().cpu().tolist(),
        "bbox_format": "xywh",
        "bbox_normalized": bool(
            batch["bboxes"].numel() == 0
            or ((batch["bboxes"] >= 0).all() and (batch["bboxes"] <= 1).all())
        ),
        "im_file_sample": jsonable(batch.get("im_file", [])[:2]),
        "ori_shape_sample": jsonable(batch.get("ori_shape", [])[:2]),
        "resized_shape_sample": jsonable(batch.get("resized_shape", [])[:2]),
        "ratio_pad_sample": jsonable(batch.get("ratio_pad", [])[:2]),
        "preprocess_method": "DetectionTrainer.preprocess_batch",
    }
    result["batch"] = batch_info

    teacher_cache: dict[int, list[Any]] = {i: [] for i in LAYERS}
    student_cache: dict[int, list[Any]] = {i: [] for i in LAYERS}
    handles = []

    def capture(cache: dict[int, list[Any]], index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            cache[index].append(output)

        return hook

    for index in LAYERS:
        handles.append(teacher_container[index].register_forward_hook(capture(teacher_cache, index)))
        handles.append(student_container[index].register_forward_hook(capture(student_cache, index)))

    student_digest_before = parameter_digest(student)
    teacher_digest_before = parameter_digest(teacher)
    try:
        with autocast(True):
            with torch.no_grad():
                teacher_prediction = teacher(batch["img"])
            standard_loss, loss_items = student(batch)

        teacher_features = {i: tensor_output(teacher_cache[i][-1]) for i in LAYERS}
        student_features = {i: tensor_output(student_cache[i][-1]) for i in LAYERS}
        shape_match = {
            str(i): list(teacher_features[i].shape) == list(student_features[i].shape) for i in LAYERS
        }
        result["features"] = {
            "teacher": {
                str(i): {
                    "shape": list(teacher_features[i].shape),
                    "dtype": str(teacher_features[i].dtype),
                    "device": str(teacher_features[i].device),
                    "requires_grad": teacher_features[i].requires_grad,
                    "hook_output_type": type(teacher_cache[i][-1]).__name__,
                    "hook_trigger_count": len(teacher_cache[i]),
                }
                for i in LAYERS
            },
            "student": {
                str(i): {
                    "shape": list(student_features[i].shape),
                    "dtype": str(student_features[i].dtype),
                    "device": str(student_features[i].device),
                    "requires_grad": student_features[i].requires_grad,
                    "hook_output_type": type(student_cache[i][-1]).__name__,
                    "hook_trigger_count": len(student_cache[i]),
                }
                for i in LAYERS
            },
            "shape_match": shape_match,
            "all_shapes_match": all(shape_match.values()),
            "teacher_prediction_type": type(teacher_prediction).__name__,
        }
        result["standard_loss"] = {
            "loss_shape": list(standard_loss.shape),
            "loss_dtype": str(standard_loss.dtype),
            "loss_requires_grad": standard_loss.requires_grad,
            "loss_values": standard_loss.detach().float().cpu().tolist(),
            "loss_sum": float(standard_loss.detach().sum()),
            "loss_items_shape": list(loss_items.shape),
            "loss_items_dtype": str(loss_items.dtype),
            "loss_items_requires_grad": loss_items.requires_grad,
            "loss_items_values": loss_items.detach().float().cpu().tolist(),
            "format": "loss: Tensor[3] (box, cls, dfl; each already multiplied by batch size); "
            "loss_items: detached Tensor[3] (unmultiplied components); trainer uses loss.sum()",
            "optimizer_step_executed": False,
            "backward_executed": False,
        }
        result["weight_integrity"] = {
            "student_parameter_sha256_before": student_digest_before,
            "student_parameter_sha256_after": parameter_digest(student),
            "teacher_parameter_sha256_before": teacher_digest_before,
            "teacher_parameter_sha256_after": parameter_digest(teacher),
        }
        result["weight_integrity"]["student_parameters_unchanged"] = (
            result["weight_integrity"]["student_parameter_sha256_before"]
            == result["weight_integrity"]["student_parameter_sha256_after"]
        )
        result["weight_integrity"]["teacher_parameters_unchanged"] = (
            result["weight_integrity"]["teacher_parameter_sha256_before"]
            == result["weight_integrity"]["teacher_parameter_sha256_after"]
        )
        result["status"] = "SUCCESS"
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()
        teacher_cache.clear()
        student_cache.clear()
        result["hook_cleanup"] = {
            "handles_removed": True,
            "teacher_cache_cleared": not teacher_cache,
            "student_cache_cleared": not student_cache,
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(jsonable(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Teacher path: {TEACHER_PT}")
    print(f"Student YAML path: {STUDENT_YAML}")
    print(f"Student best.pt path: {STUDENT_PT}")
    print(f"Data YAML path: {DATA_YAML}")
    print("Teacher P3/P4/P5 shapes:", [result["features"]["teacher"][str(i)]["shape"] for i in LAYERS])
    print("Student P3/P4/P5 shapes:", [result["features"]["student"][str(i)]["shape"] for i in LAYERS])
    print("Student standard loss:", result["standard_loss"]["format"])
    print("Batch keys:", result["batch"]["keys"])
    print("Diagnostic JSON:", OUTPUT_JSON)
    print("Run status:", result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
