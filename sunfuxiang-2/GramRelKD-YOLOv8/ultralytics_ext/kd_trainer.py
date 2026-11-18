"""Single-GPU relation-KD trainer isolated from Ultralytics core source files."""

from __future__ import annotations

import csv
import gc
import hashlib
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import distributed as dist

import ultralytics
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER, RANK, TQDM, colorstr
from ultralytics.utils.torch_utils import autocast, unset_deterministic, unwrap_model

from .feature_hooks import FeatureHookBank
from .relation_kd import ChannelRelationKDLoss

EXPECTED_ULTRALYTICS_VERSION = "8.4.84"


@dataclass
class KDConfig:
    """Configuration kept outside Ultralytics cfg parsing."""

    teacher_path: str
    layer_indices: tuple[int, ...] = (15, 18, 21)
    layer_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    pool_size: int = 20
    kd_weight: float = 1.0
    warmup_epochs: float = 10.0
    eps: float = 1e-6
    center: bool = True
    log_dir: str | None = None

    def __post_init__(self) -> None:
        self.layer_indices = tuple(int(i) for i in self.layer_indices)
        self.layer_weights = tuple(float(w) for w in self.layer_weights)
        self.pool_size = int(self.pool_size)
        self.kd_weight = float(self.kd_weight)
        self.warmup_epochs = float(self.warmup_epochs)
        self.eps = float(self.eps)
        if not Path(self.teacher_path).is_file():
            raise FileNotFoundError(f"Teacher checkpoint not found: {self.teacher_path}")
        if self.kd_weight < 0:
            raise ValueError("kd_weight must be non-negative")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")


class KDDetectionTrainer(DetectionTrainer):
    """DetectionTrainer with a frozen, optimizer-isolated Teacher and relation KD."""

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
        kd_config: KDConfig | dict[str, Any] | None = None,
    ) -> None:
        if ultralytics.__version__ != EXPECTED_ULTRALYTICS_VERSION:
            raise RuntimeError(
                f"KD trainer is locked to Ultralytics {EXPECTED_ULTRALYTICS_VERSION}, "
                f"found {ultralytics.__version__}"
            )
        if kd_config is None:
            raise ValueError("kd_config is required")
        self.kd_config = kd_config if isinstance(kd_config, KDConfig) else KDConfig(**kd_config)
        self.teacher_model: torch.nn.Module | None = None
        self.teacher_hooks: FeatureHookBank | None = None
        self.student_hooks: FeatureHookBank | None = None
        self.relation_loss: ChannelRelationKDLoss | None = None
        self.teacher_hash_before: str | None = None
        self.teacher_hash_after: str | None = None
        self.teacher_hash_unchanged: bool | None = None
        self.student_hash_before_training: str | None = None
        self.student_hash_after_training: str | None = None
        self.student_parameters_changed: bool | None = None
        self.teacher_eval_always = True
        self.kd_hooks_removed = False
        self.kd_version_lock_ok = True
        self.kd_isolation_report: dict[str, Any] = {}
        self.kd_epoch_records: list[dict[str, float]] = []
        self.kd_peak_cuda_memory_mb = 0.0
        self.kd_last_step: dict[str, Any] = {}
        self._kd_closed = False
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        if self.world_size > 1 or RANK != -1:
            raise NotImplementedError("KDDetectionTrainer currently supports single-GPU training only")
        if bool(self.args.compile):
            raise NotImplementedError("KDDetectionTrainer does not support torch.compile")
        if getattr(self.args, "distill_model", None) is not None:
            raise ValueError("Do not pass distill_model; the existing DistillationModel path is intentionally unused")

    @staticmethod
    def _module_sha256(model: torch.nn.Module) -> str:
        digest = hashlib.sha256()
        with torch.no_grad():
            for name, tensor in model.state_dict().items():
                digest.update(name.encode("utf-8"))
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _layer_output_channels(model: torch.nn.Module, index: int) -> int:
        module = unwrap_model(model).model[index]
        conv = getattr(getattr(module, "cv2", None), "conv", None)
        channels = getattr(conv, "out_channels", None)
        if channels is None:
            raise TypeError(f"Cannot determine output channels for layer {index} ({type(module).__name__})")
        return int(channels)

    def _setup_train(self) -> None:
        """Build the Student pipeline first, then load the isolated Teacher."""
        if self.world_size > 1 or RANK != -1:
            raise NotImplementedError("KDDetectionTrainer currently supports single-GPU training only")
        super()._setup_train()
        if self.world_size > 1 or RANK != -1:
            raise NotImplementedError("DDP became active unexpectedly")

        teacher, _ = load_checkpoint(self.kd_config.teacher_path)
        self.teacher_model = teacher.float().to(self.device)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)

        self.relation_loss = ChannelRelationKDLoss(
            layer_indices=self.kd_config.layer_indices,
            layer_weights=self.kd_config.layer_weights,
            pool_size=self.kd_config.pool_size,
            eps=self.kd_config.eps,
            center=self.kd_config.center,
        ).to(self.device)
        self.teacher_hooks = FeatureHookBank(
            self.teacher_model,
            layer_indices=self.kd_config.layer_indices,
            detach=True,
            name="teacher",
        )
        self.student_hooks = FeatureHookBank(
            self.model,
            layer_indices=self.kd_config.layer_indices,
            detach=False,
            name="student",
        )
        self.teacher_hooks.disable()
        self.student_hooks.disable()

        teacher_ids = {id(p) for p in self.teacher_model.parameters()}
        student_ids = {id(p) for p in unwrap_model(self.model).parameters()}
        optimizer_ids = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        ema_keys = list(self.ema.ema.state_dict()) if self.ema else []
        student_keys = list(unwrap_model(self.model).state_dict())
        channel_pairs = {
            str(index): {
                "teacher": self._layer_output_channels(self.teacher_model, index),
                "student": self._layer_output_channels(self.model, index),
            }
            for index in self.kd_config.layer_indices
        }

        if any(p.requires_grad for p in self.teacher_model.parameters()):
            raise AssertionError("Teacher has trainable parameters")
        if teacher_ids & optimizer_ids:
            raise AssertionError("Teacher parameters entered the Student optimizer")
        if teacher_ids & student_ids:
            raise AssertionError("Teacher and Student share parameter objects")
        if any("teacher" in key.lower() for key in ema_keys):
            raise AssertionError("Teacher keys entered EMA")
        if any("teacher" in key.lower() for key in student_keys):
            raise AssertionError("Teacher keys entered Student state_dict")
        if any(pair["teacher"] != pair["student"] for pair in channel_pairs.values()):
            raise AssertionError(f"Teacher/Student feature channel mismatch: {channel_pairs}")

        self.teacher_hash_before = self._module_sha256(self.teacher_model)
        self.student_hash_before_training = self._module_sha256(unwrap_model(self.model))
        self.kd_isolation_report = {
            "ultralytics_version": ultralytics.__version__,
            "version_lock_ok": True,
            "teacher_path": self.kd_config.teacher_path,
            "teacher_requires_grad_all_false": True,
            "teacher_optimizer_parameter_intersection": len(teacher_ids & optimizer_ids),
            "teacher_student_parameter_intersection": len(teacher_ids & student_ids),
            "teacher_absent_from_optimizer": not bool(teacher_ids & optimizer_ids),
            "teacher_absent_from_ema": not any("teacher" in key.lower() for key in ema_keys),
            "teacher_absent_from_student_state_dict": not any("teacher" in key.lower() for key in student_keys),
            "feature_channels": channel_pairs,
            "student_parameter_count": sum(p.numel() for p in unwrap_model(self.model).parameters()),
            "teacher_parameter_count": sum(p.numel() for p in self.teacher_model.parameters()),
            "kd_trainable_parameter_count": sum(p.numel() for p in self.relation_loss.parameters()),
            "kd_config": asdict(self.kd_config),
        }
        self._prepare_kd_metrics_file()

    def _prepare_kd_metrics_file(self) -> None:
        log_dir = Path(self.kd_config.log_dir) if self.kd_config.log_dir else Path(self.save_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.kd_metrics_path = log_dir / "kd_metrics.csv"
        if not self.args.resume:
            with self.kd_metrics_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self._kd_metric_fields()).writeheader()

    @staticmethod
    def _kd_metric_fields() -> list[str]:
        return [
            "epoch",
            "kd_weight_base",
            "kd_ramp_end",
            "det_loss_mean",
            "kd_raw_mean",
            "kd_scaled_mean",
            "kd_to_det_ratio",
            "p3_relation_loss",
            "p4_relation_loss",
            "p5_relation_loss",
            "teacher_forward_ms",
            "relation_loss_ms",
            "peak_cuda_memory_mb",
        ]

    def clear_step_features(self) -> None:
        for bank in (self.teacher_hooks, self.student_hooks):
            if bank is not None:
                bank.disable()
                bank.clear()

    def _enable_step_features(self) -> None:
        self.clear_step_features()
        if self.teacher_hooks is None or self.student_hooks is None:
            raise RuntimeError("KD hooks are not initialized")
        self.teacher_hooks.enable()
        self.student_hooks.enable()

    def compute_kd_step(
        self,
        batch: dict[str, Any],
        *,
        epoch: int,
        batch_index: int,
        number_of_batches: int,
    ) -> dict[str, Any]:
        """Run Teacher/Student forward and build the correctly scaled total loss."""
        if self.teacher_model is None or self.relation_loss is None:
            raise RuntimeError("KD runtime has not been initialized")
        if number_of_batches <= 0:
            raise ValueError("number_of_batches must be positive")
        self._enable_step_features()
        self.teacher_model.eval()

        teacher_start = torch.cuda.Event(enable_timing=True)
        teacher_end = torch.cuda.Event(enable_timing=True)
        relation_start = torch.cuda.Event(enable_timing=True)
        relation_end = torch.cuda.Event(enable_timing=True)

        with autocast(self.amp):
            teacher_start.record()
            with torch.no_grad():
                self.teacher_model(batch["img"])
            teacher_end.record()
            loss_vec, loss_items = self.model(batch)
            det_loss = loss_vec.sum()

        self.teacher_hooks.validate_once()
        self.student_hooks.validate_once()
        teacher_features = self.teacher_hooks.get()
        student_features = self.student_hooks.get()
        relation_start.record()
        kd_raw, kd_parts = self.relation_loss(student_features, teacher_features)
        relation_end.record()
        relation_end.synchronize()

        progress_epoch = float(epoch) + (float(batch_index) + 1.0) / float(number_of_batches)
        kd_ramp = (
            1.0
            if self.kd_config.warmup_epochs <= 0
            else min(1.0, progress_epoch / self.kd_config.warmup_epochs)
        )
        batch_size = int(batch["img"].shape[0])
        kd_scaled = kd_raw * self.kd_config.kd_weight * kd_ramp * batch_size
        total_loss = det_loss + kd_scaled
        if not torch.isfinite(total_loss):
            raise FloatingPointError("total KD training loss contains NaN or Inf")

        result = {
            "loss_vec": loss_vec,
            "loss_items": loss_items,
            "det_loss": det_loss,
            "kd_raw": kd_raw,
            "kd_parts": kd_parts,
            "kd_ramp": kd_ramp,
            "kd_scaled": kd_scaled,
            "total_loss": total_loss,
            "batch_size": batch_size,
            "teacher_forward_ms": teacher_start.elapsed_time(teacher_end),
            "relation_loss_ms": relation_start.elapsed_time(relation_end),
            "teacher_hook_counts": dict(self.teacher_hooks.counts),
            "student_hook_counts": dict(self.student_hooks.counts),
            "teacher_eval": not self.teacher_model.training,
        }
        self.kd_last_step = {
            "kd_ramp": kd_ramp,
            "teacher_hook_counts": result["teacher_hook_counts"],
            "student_hook_counts": result["student_hook_counts"],
            "teacher_eval": result["teacher_eval"],
        }
        self.teacher_eval_always = self.teacher_eval_always and bool(result["teacher_eval"])
        return result

    def _reset_kd_epoch_stats(self) -> None:
        self._kd_epoch_count = 0
        self._kd_epoch_sums = {
            key: 0.0
            for key in (
                "det_loss",
                "kd_raw",
                "kd_scaled",
                "ratio",
                "p3",
                "p4",
                "p5",
                "teacher_ms",
                "relation_ms",
            )
        }
        self._kd_ramp_last = 0.0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _update_kd_epoch_stats(self, step: dict[str, Any]) -> None:
        det = float(step["det_loss"].detach())
        scaled = float(step["kd_scaled"].detach())
        parts = step["kd_parts"]
        self._kd_epoch_count += 1
        self._kd_epoch_sums["det_loss"] += det
        self._kd_epoch_sums["kd_raw"] += float(step["kd_raw"].detach())
        self._kd_epoch_sums["kd_scaled"] += scaled
        self._kd_epoch_sums["ratio"] += scaled / max(abs(det), 1e-12)
        self._kd_epoch_sums["p3"] += float(parts["p3"].detach())
        self._kd_epoch_sums["p4"] += float(parts["p4"].detach())
        self._kd_epoch_sums["p5"] += float(parts["p5"].detach())
        self._kd_epoch_sums["teacher_ms"] += float(step["teacher_forward_ms"])
        self._kd_epoch_sums["relation_ms"] += float(step["relation_loss_ms"])
        self._kd_ramp_last = float(step["kd_ramp"])

    def _write_kd_epoch_metrics(self, epoch: int) -> None:
        if self._kd_epoch_count <= 0:
            return
        n = float(self._kd_epoch_count)
        peak_mb = (
            torch.cuda.max_memory_allocated(self.device) / (1024**2) if self.device.type == "cuda" else 0.0
        )
        self.kd_peak_cuda_memory_mb = max(self.kd_peak_cuda_memory_mb, peak_mb)
        row = {
            "epoch": int(epoch + 1),
            "kd_weight_base": self.kd_config.kd_weight,
            "kd_ramp_end": self._kd_ramp_last,
            "det_loss_mean": self._kd_epoch_sums["det_loss"] / n,
            "kd_raw_mean": self._kd_epoch_sums["kd_raw"] / n,
            "kd_scaled_mean": self._kd_epoch_sums["kd_scaled"] / n,
            "kd_to_det_ratio": self._kd_epoch_sums["ratio"] / n,
            "p3_relation_loss": self._kd_epoch_sums["p3"] / n,
            "p4_relation_loss": self._kd_epoch_sums["p4"] / n,
            "p5_relation_loss": self._kd_epoch_sums["p5"] / n,
            "teacher_forward_ms": self._kd_epoch_sums["teacher_ms"] / n,
            "relation_loss_ms": self._kd_epoch_sums["relation_ms"] / n,
            "peak_cuda_memory_mb": peak_mb,
        }
        with self.kd_metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self._kd_metric_fields()).writerow(row)
        self.kd_epoch_records.append(row)
        LOGGER.info(
            "KD: "
            f"raw={row['kd_raw_mean']:.6g} scaled={row['kd_scaled_mean']:.6g} "
            f"ratio={row['kd_to_det_ratio']:.4f} p3={row['p3_relation_loss']:.6g} "
            f"p4={row['p4_relation_loss']:.6g} p5={row['p5_relation_loss']:.6g} "
            f"ramp={row['kd_ramp_end']:.4f}"
        )

    def close_kd(self) -> None:
        """Remove hooks, record Teacher integrity, and release Teacher CUDA storage."""
        if self._kd_closed:
            return
        self.clear_step_features()
        if self.teacher_hooks is not None:
            self.teacher_hooks.remove()
        if self.student_hooks is not None:
            self.student_hooks.remove()
        self.kd_hooks_removed = bool(
            (self.teacher_hooks is None or self.teacher_hooks.removed)
            and (self.student_hooks is None or self.student_hooks.removed)
        )
        if self.teacher_model is not None:
            self.teacher_hash_after = self._module_sha256(self.teacher_model)
            self.teacher_hash_unchanged = self.teacher_hash_before == self.teacher_hash_after
            self.kd_isolation_report.update(
                {
                    "teacher_hash_before": self.teacher_hash_before,
                    "teacher_hash_after": self.teacher_hash_after,
                    "teacher_hash_unchanged": self.teacher_hash_unchanged,
                    "hooks_removed": self.kd_hooks_removed,
                }
            )
        if getattr(self, "model", None) is not None:
            self.student_hash_after_training = self._module_sha256(unwrap_model(self.model))
            self.student_parameters_changed = (
                self.student_hash_before_training != self.student_hash_after_training
            )
            self.kd_isolation_report.update(
                {
                    "student_hash_before_training": self.student_hash_before_training,
                    "student_hash_after_training": self.student_hash_after_training,
                    "student_parameters_changed": self.student_parameters_changed,
                    "teacher_eval_always": self.teacher_eval_always,
                }
            )
        self.teacher_model = None
        self.teacher_hooks = None
        self.student_hooks = None
        self._kd_closed = True
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _do_train(self) -> None:
        """Run the local 8.4.84 loop with relation KD inserted before backward.

        Copied from this checkout's ``ultralytics/engine/trainer.py:390-603``.
        Only the original forward/backward block at lines 455-473 is extended.
        """
        try:
            if self.world_size > 1:
                raise NotImplementedError("KDDetectionTrainer currently supports single-GPU training only")
            self._setup_train()

            nb = len(self.train_loader)  # number of batches
            nw = (
                max(round(self.args.warmup_epochs * nb), 100)
                if self.args.warmup_epochs > 0
                else -1
            )  # warmup iterations
            last_opt_step = -1
            self.epoch_time = None
            self.epoch_time_start = time.time()
            self.train_time_start = time.time()
            self.run_callbacks("on_train_start")
            LOGGER.info(
                f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
                f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
                f"Logging results to {colorstr('bold', self.save_dir)}\n"
                f"Starting training for "
                + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
            )
            if self.args.close_mosaic:
                base_idx = (self.epochs - self.args.close_mosaic) * nb
                self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
            epoch = self.start_epoch
            self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
            self._oom_retries = 0  # OOM auto-reduce counter for first epoch
            while True:
                self.epoch = epoch
                self._reset_kd_epoch_stats()
                self.run_callbacks("on_train_epoch_start")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # suppress scheduler warning
                    self.scheduler.step()

                self._model_train()
                self.teacher_model.eval()
                if RANK != -1:
                    self.train_loader.sampler.set_epoch(epoch)
                pbar = enumerate(self.train_loader)
                if epoch == (self.epochs - self.args.close_mosaic):
                    self._close_dataloader_mosaic()
                    self.train_loader.reset()

                if RANK in {-1, 0}:
                    LOGGER.info(self.progress_string())
                    pbar = TQDM(enumerate(self.train_loader), total=nb)
                self.tloss = None
                for i, batch in pbar:
                    self.run_callbacks("on_train_batch_start")
                    ni = i + nb * epoch
                    if ni <= nw:
                        xi = [0, nw]
                        self.accumulate = max(
                            1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                        )
                        for x in self.optimizer.param_groups:
                            x["lr"] = np.interp(
                                ni,
                                xi,
                                [
                                    self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                    x["initial_lr"] * self.lf(epoch),
                                ],
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(
                                    ni, xi, [self.args.warmup_momentum, self.args.momentum]
                                )

                    # KD extension of local trainer.py:455-473.
                    try:
                        try:
                            with autocast(self.amp):
                                batch = self.preprocess_batch(batch)
                            kd_step = self.compute_kd_step(
                                batch,
                                epoch=epoch,
                                batch_index=i,
                                number_of_batches=nb,
                            )
                            loss = kd_step["loss_vec"]
                            self.loss_items = kd_step["loss_items"]
                            self.loss = kd_step["total_loss"]
                            if RANK != -1:
                                self.loss *= self.world_size
                            self.tloss = (
                                self.loss_items
                                if self.tloss is None
                                else (self.tloss * i + self.loss_items) / (i + 1)
                            )
                            self.scaler.scale(self.loss).backward()
                            self._update_kd_epoch_stats(kd_step)
                        finally:
                            # Student features hold the graph until backward completes.
                            self.clear_step_features()
                    except RuntimeError as e:
                        is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
                        if not is_oom and not any(
                            s in str(e)
                            for s in ("CUDNN_STATUS_INTERNAL_ERROR", "unable to find an engine")
                        ):
                            raise
                        if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:
                            raise
                        self._oom_retries += 1
                        old_batch = self.batch_size
                        self.args.batch = self.batch_size = max(self.batch_size // 2, 1)
                        LOGGER.warning(
                            f"{'CUDA out of memory' if is_oom else 'CUDA backend memory error'} "
                            f"with batch={old_batch}. Reducing to batch={self.batch_size} "
                            f"and retrying ({self._oom_retries}/3)."
                        )
                        batch = loss = None
                        self.loss = self.loss_items = self.tloss = None
                        self._clear_memory()
                        self._build_train_pipeline()
                        self.scheduler.last_epoch = self.start_epoch - 1
                        nb = len(self.train_loader)
                        nw = (
                            max(round(self.args.warmup_epochs * nb), 100)
                            if self.args.warmup_epochs > 0
                            else -1
                        )
                        last_opt_step = -1
                        self.optimizer.zero_grad()
                        break
                    if ni - last_opt_step >= self.accumulate:
                        self.optimizer_step()
                        last_opt_step = ni

                        if self.args.time:
                            self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                            if RANK != -1:
                                broadcast_list = [self.stop if RANK == 0 else None]
                                dist.broadcast_object_list(broadcast_list, 0)
                                self.stop = broadcast_list[0]
                            if self.stop:
                                break

                    if RANK in {-1, 0}:
                        loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                        pbar.set_description(
                            ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                            % (
                                f"{epoch + 1}/{self.epochs}",
                                f"{self._get_memory():.3g}G",
                                *(
                                    self.tloss
                                    if loss_length > 1
                                    else torch.unsqueeze(self.tloss, 0)
                                ),
                                batch.get("cls", batch["img"]).shape[0],
                                batch["img"].shape[-1],
                            )
                        )
                        self.run_callbacks("on_batch_end")
                        if self.args.plots and ni in self.plot_idx:
                            self.plot_training_samples(batch, ni)

                    self.run_callbacks("on_train_batch_end")
                    if self.stop:
                        break
                else:
                    self._oom_retries = 0

                if self._oom_retries and not self.stop:
                    continue

                self._write_kd_epoch_metrics(epoch)
                if hasattr(unwrap_model(self.model).criterion, "update"):
                    unwrap_model(self.model).criterion.update()

                self.lr = {
                    f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)
                }

                self.run_callbacks("on_train_epoch_end")
                if RANK in {-1, 0}:
                    self.ema.update_attr(
                        self.model,
                        include=["yaml", "nc", "args", "names", "stride", "class_weights"],
                    )

                final_epoch = epoch + 1 >= self.epochs
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self._clear_memory(None if self.device.type == "mps" else 0.5)
                    self.metrics, self.fitness = self.validate()

                if self._handle_nan_recovery(epoch):
                    continue

                self.nan_recovery_attempts = 0
                if RANK in {-1, 0}:
                    self.save_metrics(
                        metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr}
                    )
                    self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                    if self.args.time:
                        self.stop |= (time.time() - self.train_time_start) > (
                            self.args.time * 3600
                        )

                    if (self.args.save or final_epoch) and self.save_model():
                        self.run_callbacks("on_model_save")

                t = time.time()
                self.epoch_time = t - self.epoch_time_start
                self.epoch_time_start = t
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (
                        epoch - self.start_epoch + 1
                    )
                    self.epochs = self.args.epochs = math.ceil(
                        self.args.time * 3600 / mean_epoch_time
                    )
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch
                    self.stop |= epoch >= self.epochs
                self.run_callbacks("on_fit_epoch_end")
                self._clear_memory(None if self.device.type == "mps" else 0.5)

                if RANK != -1:
                    broadcast_list = [self.stop if RANK == 0 else None]
                    dist.broadcast_object_list(broadcast_list, 0)
                    self.stop = broadcast_list[0]
                if self.stop:
                    break
                epoch += 1

            seconds = time.time() - self.train_time_start
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in "
                f"{seconds / 3600:.3f} hours."
            )
            self.final_eval()
            if RANK in {-1, 0}:
                if self.args.plots:
                    self.plot_metrics()
                self.run_callbacks("on_train_end")
            self._clear_memory()
            unset_deterministic()
            self.run_callbacks("teardown")
        finally:
            self.close_kd()
