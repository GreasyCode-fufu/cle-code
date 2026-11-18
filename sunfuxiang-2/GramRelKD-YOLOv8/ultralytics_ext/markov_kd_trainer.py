"""MTRD trainer that reuses the verified Gram-RelKD training infrastructure."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.utils import LOGGER

from .kd_trainer import KDConfig, KDDetectionTrainer
from .markov_relation_kd import MultiStepTransitionKDLoss


@dataclass
class MTRDConfig:
    """Configuration kept outside Ultralytics cfg parsing."""

    teacher_path: str
    layer_indices: tuple[int, ...] = (15, 18, 21)
    layer_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    pool_size: int = 20
    temperature: float = 0.2
    transition_steps: tuple[int, ...] = (1, 2)
    step_weights: tuple[float, ...] = (1.0, 1.0)
    self_loop_prob: float = 0.0
    divergence: str = "kl"
    kd_weight: float = 1.0
    warmup_epochs: float = 10.0
    eps: float = 1e-6
    center: bool = True
    log_dir: str | None = None

    def __post_init__(self) -> None:
        self.layer_indices = tuple(int(i) for i in self.layer_indices)
        self.layer_weights = tuple(float(w) for w in self.layer_weights)
        self.pool_size = int(self.pool_size)
        self.temperature = float(self.temperature)
        self.transition_steps = tuple(int(step) for step in self.transition_steps)
        self.step_weights = tuple(float(w) for w in self.step_weights)
        self.self_loop_prob = float(self.self_loop_prob)
        self.kd_weight = float(self.kd_weight)
        self.warmup_epochs = float(self.warmup_epochs)
        self.eps = float(self.eps)
        if not Path(self.teacher_path).is_file():
            raise FileNotFoundError(f"Teacher checkpoint not found: {self.teacher_path}")
        if self.kd_weight < 0:
            raise ValueError("kd_weight must be non-negative")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        # Reuse the loss constructor as the single authoritative config validator.
        MultiStepTransitionKDLoss(
            layer_indices=self.layer_indices,
            layer_weights=self.layer_weights,
            pool_size=self.pool_size,
            temperature=self.temperature,
            transition_steps=self.transition_steps,
            step_weights=self.step_weights,
            self_loop_prob=self.self_loop_prob,
            divergence=self.divergence,
            center=self.center,
            eps=self.eps,
        )


class MarkovKDDetectionTrainer(KDDetectionTrainer):
    """KDDetectionTrainer with only the relation objective and metrics replaced by MTRD."""

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
        mtrd_config: MTRDConfig | dict[str, Any] | None = None,
    ) -> None:
        if mtrd_config is None:
            raise ValueError("mtrd_config is required")
        self.mtrd_config = (
            mtrd_config if isinstance(mtrd_config, MTRDConfig) else MTRDConfig(**mtrd_config)
        )
        self.mtrd_batch_memory_allocated_mb: list[float] = []
        base_config = KDConfig(
            teacher_path=self.mtrd_config.teacher_path,
            layer_indices=self.mtrd_config.layer_indices,
            layer_weights=self.mtrd_config.layer_weights,
            pool_size=self.mtrd_config.pool_size,
            kd_weight=self.mtrd_config.kd_weight,
            warmup_epochs=self.mtrd_config.warmup_epochs,
            eps=self.mtrd_config.eps,
            center=self.mtrd_config.center,
            log_dir=self.mtrd_config.log_dir,
        )
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks, kd_config=base_config)

    def compute_kd_step(
        self,
        batch: dict[str, Any],
        *,
        epoch: int,
        batch_index: int,
        number_of_batches: int,
    ) -> dict[str, Any]:
        result = super().compute_kd_step(
            batch,
            epoch=epoch,
            batch_index=batch_index,
            number_of_batches=number_of_batches,
        )
        if self.device.type == "cuda":
            self.mtrd_batch_memory_allocated_mb.append(
                torch.cuda.memory_allocated(self.device) / (1024**2)
            )
        return result

    def _setup_train(self) -> None:
        super()._setup_train()
        self.relation_loss = MultiStepTransitionKDLoss(
            layer_indices=self.mtrd_config.layer_indices,
            layer_weights=self.mtrd_config.layer_weights,
            pool_size=self.mtrd_config.pool_size,
            temperature=self.mtrd_config.temperature,
            transition_steps=self.mtrd_config.transition_steps,
            step_weights=self.mtrd_config.step_weights,
            self_loop_prob=self.mtrd_config.self_loop_prob,
            divergence=self.mtrd_config.divergence,
            center=self.mtrd_config.center,
            eps=self.mtrd_config.eps,
        ).to(self.device)
        self.kd_isolation_report.update(
            {
                "kd_method": "MTRD",
                "mtrd_config": asdict(self.mtrd_config),
                "kd_trainable_parameter_count": sum(
                    parameter.numel() for parameter in self.relation_loss.parameters()
                ),
            }
        )

    def _prepare_kd_metrics_file(self) -> None:
        log_dir = Path(self.kd_config.log_dir) if self.kd_config.log_dir else Path(self.save_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.kd_metrics_path = log_dir / "mtrd_metrics.csv"
        self.mtrd_metrics_path = self.kd_metrics_path
        if not self.args.resume:
            with self.kd_metrics_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self._kd_metric_fields()).writeheader()

    @staticmethod
    def _kd_metric_fields() -> list[str]:
        return [
            "epoch",
            "temperature",
            "self_loop_prob",
            "kd_weight_base",
            "kd_ramp_end",
            "det_loss_mean",
            "mtrd_raw_mean",
            "mtrd_scaled_mean",
            "mtrd_to_det_ratio",
            "p3_step1",
            "p3_step2",
            "p4_step1",
            "p4_step2",
            "p5_step1",
            "p5_step2",
            "p3_entropy",
            "p4_entropy",
            "p5_entropy",
            "p3_max_prob",
            "p4_max_prob",
            "p5_max_prob",
            "teacher_forward_ms",
            "transition_loss_ms",
            "peak_cuda_memory_mb",
        ]

    def _reset_kd_epoch_stats(self) -> None:
        self._kd_epoch_count = 0
        self._kd_epoch_sums = {
            key: 0.0
            for key in (
                "det_loss",
                "mtrd_raw",
                "mtrd_scaled",
                "ratio",
                "p3_step1",
                "p3_step2",
                "p4_step1",
                "p4_step2",
                "p5_step1",
                "p5_step2",
                "p3_entropy",
                "p4_entropy",
                "p5_entropy",
                "p3_max_prob",
                "p4_max_prob",
                "p5_max_prob",
                "teacher_ms",
                "transition_ms",
            )
        }
        self._kd_ramp_last = 0.0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    @staticmethod
    def _part(parts: dict[str, torch.Tensor], key: str) -> float:
        value = parts.get(key)
        return 0.0 if value is None else float(value.detach())

    def _update_kd_epoch_stats(self, step: dict[str, Any]) -> None:
        det = float(step["det_loss"].detach())
        scaled = float(step["kd_scaled"].detach())
        parts = step["kd_parts"]
        self._kd_epoch_count += 1
        self._kd_epoch_sums["det_loss"] += det
        self._kd_epoch_sums["mtrd_raw"] += float(step["kd_raw"].detach())
        self._kd_epoch_sums["mtrd_scaled"] += scaled
        self._kd_epoch_sums["ratio"] += scaled / max(abs(det), 1e-12)
        for name in ("p3", "p4", "p5"):
            self._kd_epoch_sums[f"{name}_step1"] += self._part(parts, f"{name}_step1")
            self._kd_epoch_sums[f"{name}_step2"] += self._part(parts, f"{name}_step2")
            self._kd_epoch_sums[f"{name}_entropy"] += self._part(
                parts, f"transition_entropy_{name}"
            )
            self._kd_epoch_sums[f"{name}_max_prob"] += self._part(
                parts, f"transition_max_prob_{name}"
            )
        self._kd_epoch_sums["teacher_ms"] += float(step["teacher_forward_ms"])
        self._kd_epoch_sums["transition_ms"] += float(step["relation_loss_ms"])
        self._kd_ramp_last = float(step["kd_ramp"])

    def _write_kd_epoch_metrics(self, epoch: int) -> None:
        if self._kd_epoch_count <= 0:
            return
        n = float(self._kd_epoch_count)
        peak_mb = (
            torch.cuda.max_memory_allocated(self.device) / (1024**2)
            if self.device.type == "cuda"
            else 0.0
        )
        self.kd_peak_cuda_memory_mb = max(self.kd_peak_cuda_memory_mb, peak_mb)
        row = {
            "epoch": int(epoch + 1),
            "temperature": self.mtrd_config.temperature,
            "self_loop_prob": self.mtrd_config.self_loop_prob,
            "kd_weight_base": self.kd_config.kd_weight,
            "kd_ramp_end": self._kd_ramp_last,
            "det_loss_mean": self._kd_epoch_sums["det_loss"] / n,
            "mtrd_raw_mean": self._kd_epoch_sums["mtrd_raw"] / n,
            "mtrd_scaled_mean": self._kd_epoch_sums["mtrd_scaled"] / n,
            "mtrd_to_det_ratio": self._kd_epoch_sums["ratio"] / n,
            "teacher_forward_ms": self._kd_epoch_sums["teacher_ms"] / n,
            "transition_loss_ms": self._kd_epoch_sums["transition_ms"] / n,
            "peak_cuda_memory_mb": peak_mb,
        }
        for name in ("p3", "p4", "p5"):
            for suffix in ("step1", "step2", "entropy", "max_prob"):
                row[f"{name}_{suffix}"] = self._kd_epoch_sums[f"{name}_{suffix}"] / n
        with self.kd_metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self._kd_metric_fields()).writerow(row)
        self.kd_epoch_records.append(row)
        LOGGER.info(
            "MTRD: "
            f"raw={row['mtrd_raw_mean']:.6g} scaled={row['mtrd_scaled_mean']:.6g} "
            f"ratio={row['mtrd_to_det_ratio']:.4f} "
            f"p3=({row['p3_step1']:.5g},{row['p3_step2']:.5g}) "
            f"p4=({row['p4_step1']:.5g},{row['p4_step2']:.5g}) "
            f"p5=({row['p5_step1']:.5g},{row['p5_step2']:.5g}) "
            f"ramp={row['kd_ramp_end']:.4f}"
        )
