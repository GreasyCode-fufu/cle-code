"""One-real-batch integration test for the independent KD training pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from ultralytics.utils.torch_utils import autocast, unwrap_model
from ultralytics_ext.kd_trainer import KDConfig, KDDetectionTrainer


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "runs" / "analysis" / "kd_implementation_check"
TEACHER = ROOT / "runs" / "detect" / "runs" / "detect" / "hongkong_L1-96" / "weights" / "best.pt"
STUDENT = ROOT / "ultralytics" / "cfg" / "models" / "hongkong" / "yolov8n-L1-96-SR1.yaml"
DATA = ROOT / "ultralytics" / "cfg" / "datasets" / "coco8.yaml"
CFG = ROOT / "ultralytics" / "cfg" / "default.yaml"
EXPECTED_SHAPES = {
    15: [4, 64, 80, 80],
    18: [4, 96, 40, 40],
    21: [4, 256, 20, 20],
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real KD pipeline test requires CUDA")
def test_real_batch_backward_optimizer_and_isolation() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = ANALYSIS_DIR / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = temp_dir / "pipeline_student_only.pt"
    result: dict[str, object] = {
        "status": "started",
        "teacher_path": str(TEACHER),
        "student_yaml": str(STUDENT),
        "data_yaml": str(DATA),
    }
    trainer: KDDetectionTrainer | None = None

    try:
        kd_config = KDConfig(
            teacher_path=str(TEACHER),
            layer_indices=(15, 18, 21),
            layer_weights=(1.0, 1.0, 1.0),
            pool_size=20,
            kd_weight=1.0,
            warmup_epochs=10.0,
            log_dir=str(ANALYSIS_DIR / "pipeline"),
        )
        overrides = {
            "model": str(STUDENT),
            "data": str(DATA),
            "epochs": 1,
            "fraction": 0.02,
            "batch": 4,
            "imgsz": 640,
            "workers": 0,
            "device": "0",
            "project": str(temp_dir),
            "name": "pipeline_runtime",
            "exist_ok": True,
            "seed": 0,
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
        trainer = KDDetectionTrainer(cfg=str(CFG), overrides=overrides, kd_config=kd_config)
        trainer._setup_train()
        trainer._model_train()
        # A scale of 1 makes this a direct gradient-finiteness diagnostic while still
        # exercising the production GradScaler call path.
        trainer.scaler = torch.amp.GradScaler("cuda", enabled=trainer.amp, init_scale=1.0)
        raw_batch = next(iter(trainer.train_loader))
        with autocast(trainer.amp):
            batch = trainer.preprocess_batch(raw_batch)

        torch.cuda.reset_peak_memory_stats(trainer.device)
        teacher_hash_before = trainer._module_sha256(trainer.teacher_model)
        student_hash_before = trainer._module_sha256(unwrap_model(trainer.model))
        trainer.optimizer.zero_grad(set_to_none=True)
        step = trainer.compute_kd_step(
            batch,
            epoch=0,
            batch_index=0,
            number_of_batches=len(trainer.train_loader),
        )

        teacher_features = trainer.teacher_hooks.get()
        student_features = trainer.student_hooks.get()
        actual_shapes = {index: list(tensor.shape) for index, tensor in student_features.items()}
        assert actual_shapes == EXPECTED_SHAPES
        assert {index: list(tensor.shape) for index, tensor in teacher_features.items()} == EXPECTED_SHAPES
        assert all(not tensor.requires_grad for tensor in teacher_features.values())
        assert all(tensor.requires_grad for tensor in student_features.values())
        assert all(count == 1 for count in step["teacher_hook_counts"].values())
        assert all(count == 1 for count in step["student_hook_counts"].values())
        assert step["teacher_eval"] is True
        assert step["det_loss"].shape == torch.Size([])
        assert step["kd_raw"].shape == torch.Size([])
        assert step["kd_scaled"].shape == torch.Size([])
        assert step["total_loss"].shape == torch.Size([])
        assert all(torch.isfinite(value) for value in (step["det_loss"], step["kd_raw"], step["kd_scaled"]))

        trainer.scaler.scale(step["total_loss"]).backward()
        student_gradients = [
            parameter.grad
            for parameter in unwrap_model(trainer.model).parameters()
            if parameter.grad is not None
        ]
        assert student_gradients
        assert all(torch.isfinite(gradient).all() for gradient in student_gradients)
        assert any(torch.count_nonzero(gradient).item() > 0 for gradient in student_gradients)
        assert all(parameter.grad is None for parameter in trainer.teacher_model.parameters())

        trainer.optimizer_step()
        student_hash_after_step = trainer._module_sha256(unwrap_model(trainer.model))
        teacher_hash_after_step = trainer._module_sha256(trainer.teacher_model)
        assert student_hash_after_step != student_hash_before
        assert teacher_hash_after_step == teacher_hash_before

        optimizer_ids = {id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]}
        teacher_ids = {id(parameter) for parameter in trainer.teacher_model.parameters()}
        assert not (optimizer_ids & teacher_ids)
        assert not any("teacher" in key.lower() for key in trainer.ema.ema.state_dict())
        assert not any("teacher" in key.lower() for key in unwrap_model(trainer.model).state_dict())

        trainer.clear_step_features()
        assert all(count == 0 for count in trainer.teacher_hooks.counts.values())
        assert all(count == 0 for count in trainer.student_hooks.counts.values())
        with pytest.raises(RuntimeError, match="missing hook outputs"):
            trainer.student_hooks.get()

        # Ultralytics 8.4.84 checkpoints serialize EMA, which is created before KD
        # hooks are attached and therefore contains neither hook closures nor Teacher.
        torch.save({"model": None, "ema": copy.deepcopy(trainer.ema.ema).half().cpu()}, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_keys = list(checkpoint["ema"].state_dict())
        teacher_absent_checkpoint = not any("teacher" in key.lower() for key in checkpoint_keys)
        assert teacher_absent_checkpoint

        result.update(
            {
                "status": "passed",
                "batch_size": int(batch["img"].shape[0]),
                "feature_shapes": actual_shapes,
                "det_loss": float(step["det_loss"].detach()),
                "kd_raw": float(step["kd_raw"].detach()),
                "kd_scaled": float(step["kd_scaled"].detach()),
                "kd_ramp": float(step["kd_ramp"]),
                "total_loss": float(step["total_loss"].detach()),
                "loss_vec_shape": list(step["loss_vec"].shape),
                "student_gradients_finite": True,
                "student_nonzero_gradient": True,
                "student_parameters_changed_after_step": True,
                "teacher_gradients_all_none": True,
                "teacher_eval": bool(step["teacher_eval"]),
                "teacher_hash_before": teacher_hash_before,
                "teacher_hash_after_step": teacher_hash_after_step,
                "teacher_hash_unchanged_after_step": teacher_hash_before == teacher_hash_after_step,
                "teacher_absent_from_optimizer": not bool(optimizer_ids & teacher_ids),
                "teacher_absent_from_ema": True,
                "teacher_absent_from_student_state_dict": True,
                "teacher_absent_from_checkpoint": teacher_absent_checkpoint,
                "teacher_hook_counts": step["teacher_hook_counts"],
                "student_hook_counts": step["student_hook_counts"],
                "features_cleared_after_backward": True,
                "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(trainer.device) / (1024**2),
            }
        )
    finally:
        checkpoint_path.unlink(missing_ok=True)
        if trainer is not None:
            trainer.close_kd()
            result["hooks_removed"] = trainer.kd_hooks_removed
            result["teacher_hash_after_close"] = trainer.teacher_hash_after
            result["teacher_hash_unchanged_after_close"] = trainer.teacher_hash_unchanged
            result["teacher_eval_always"] = trainer.teacher_eval_always
            result["isolation"] = trainer.kd_isolation_report
        (ANALYSIS_DIR / "pipeline_test_results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        (ANALYSIS_DIR / "teacher_optimizer_isolation.json").write_text(
            json.dumps(result.get("isolation", {}), indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    assert result["status"] == "passed"
    assert result["hooks_removed"] is True
    assert result["teacher_hash_unchanged_after_close"] is True
