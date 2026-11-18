#!/usr/bin/env python3
"""No-training smoke test for the frozen Gram RelKD Student checkpoint."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

extension_before = any(
    name == "ultralytics_ext" or name.startswith("ultralytics_ext.") for name in sys.modules
)

from ultralytics import YOLO, __version__
from ultralytics.utils.torch_utils import get_flops

WEIGHTS = ROOT / "runs/detect/hongkong_SR1_P4Only_RelKD/weights/best.pt"
OUTPUT = ROOT / "docs/gram_relkd_handoff/CLONE_SMOKE_TEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if __version__ != "8.4.84":
        raise RuntimeError(f"Expected Ultralytics 8.4.84, got {__version__}")
    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint.get("ema") or checkpoint.get("model")
    if checkpoint_model is None:
        raise RuntimeError("Checkpoint contains neither ema nor model")

    forbidden_tokens = (
        "teacher", "relation_kd", "gram_kd", "distill", "mtrd", "markov_kd",
        "discriminator", "adversarial", "response_kd", "attention_kd", "composite_kd",
    )
    state_keys = list(checkpoint_model.state_dict().keys())
    forbidden_state_keys = [
        key for key in state_keys if any(token in key.lower() for token in forbidden_tokens)
    ]
    forbidden_top_level_keys = [
        str(key) for key in checkpoint.keys()
        if any(token in str(key).lower() for token in forbidden_tokens)
    ]
    forbidden_module_classes = sorted({
        f"{module.__class__.__module__}.{module.__class__.__name__}"
        for module in checkpoint_model.modules()
        if module.__class__.__module__.startswith("ultralytics_ext")
        or any(token in module.__class__.__name__.lower() for token in forbidden_tokens)
    })

    loaded = YOLO(str(WEIGHTS))
    model = loaded.model.float().cpu().eval()
    print(model)
    features: dict[str, torch.Tensor] = {}
    handles = []
    for index in (15, 18, 21):
        handles.append(model.model[index].register_forward_hook(
            lambda _module, _inputs, output, idx=index: features.__setitem__(str(idx), output)
        ))
    with torch.no_grad():
        prediction = model(torch.randn(1, 3, 640, 640))
    for handle in handles:
        handle.remove()

    feature_shapes = {key: list(value.shape) for key, value in features.items()}
    expected_shapes = {
        "15": [1, 64, 80, 80],
        "18": [1, 96, 40, 40],
        "21": [1, 256, 20, 20],
    }
    hidden_channels = {
        str(index): int(getattr(model.model[index], "c"))
        for index in (12, 18, 21)
    }
    expected_hidden = {"12": 40, "18": 40, "21": 128}

    unfused_params = sum(parameter.numel() for parameter in checkpoint_model.parameters())
    fused_model = copy.deepcopy(checkpoint_model).float().cpu().fuse(verbose=False)
    fused_params = sum(parameter.numel() for parameter in fused_model.parameters())
    gflops = float(get_flops(fused_model, imgsz=640))
    extension_after = any(
        name == "ultralytics_ext" or name.startswith("ultralytics_ext.") for name in sys.modules
    )

    result = {
        "status": "SUCCESS",
        "training_launched": False,
        "full_test_evaluation_launched": False,
        "project_root": str(ROOT),
        "weights": str(WEIGHTS),
        "weights_sha256": sha256(WEIGHTS),
        "ultralytics_version": __version__,
        "model_structure_printed": True,
        "model_is_sr1_p4only": feature_shapes == expected_shapes and hidden_channels == expected_hidden,
        "hidden_channels": hidden_channels,
        "expected_hidden_channels": expected_hidden,
        "feature_shapes": feature_shapes,
        "expected_feature_shapes": expected_shapes,
        "random_eval_forward_success": prediction is not None,
        "unfused_params": unfused_params,
        "fused_params": fused_params,
        "gflops": gflops,
        "checkpoint_top_level_keys": [str(key) for key in checkpoint.keys()],
        "forbidden_top_level_keys": forbidden_top_level_keys,
        "forbidden_state_dict_keys": forbidden_state_keys,
        "forbidden_module_classes": forbidden_module_classes,
        "teacher_absent": not any("teacher" in item.lower() for item in forbidden_top_level_keys + forbidden_state_keys + forbidden_module_classes),
        "kd_modules_absent": not forbidden_top_level_keys and not forbidden_state_keys and not forbidden_module_classes,
        "discriminator_absent": not any("discriminator" in item.lower() for item in forbidden_top_level_keys + forbidden_state_keys + forbidden_module_classes),
        "ultralytics_ext_loaded_before_model": extension_before,
        "ultralytics_ext_loaded_after_inference": extension_after,
        "inference_independent_of_ultralytics_ext": not extension_before and not extension_after,
    }
    checks = [
        result["model_is_sr1_p4only"], result["random_eval_forward_success"],
        result["teacher_absent"], result["kd_modules_absent"], result["discriminator_absent"],
        result["inference_independent_of_ultralytics_ext"], unfused_params == 2_607_913,
        fused_params == 2_603_129, abs(gflops - 7.044096) < 1e-9,
    ]
    result["all_checks_passed"] = all(checks)
    result["status"] = "SUCCESS" if result["all_checks_passed"] else "FAILED"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

