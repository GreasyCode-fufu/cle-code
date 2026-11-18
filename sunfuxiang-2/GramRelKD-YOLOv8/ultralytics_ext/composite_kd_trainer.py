"""Single-GPU composite KD trainer isolated from Ultralytics core."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import autocast, unwrap_model

from .adversarial_feature_kd import AdversarialFeatureKD
from .composite_kd import CompositeKDConfig, CompositeKDLoss
from .kd_trainer import KDConfig, KDDetectionTrainer
from .response_kd import DetectOutputHook


class CompositeKDDetectionTrainer(KDDetectionTrainer):
    """Reuse the verified KD loop while composing independently weighted objectives."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, composite_config=None, teacher_path=None):
        if composite_config is None or teacher_path is None:
            raise ValueError("composite_config and teacher_path are required")
        self.composite_config = composite_config if isinstance(composite_config, CompositeKDConfig) else CompositeKDConfig(**composite_config)
        self.composite_loss = None
        self.teacher_head_hook = self.student_head_hook = None
        self.adversarial = None
        self._composite_sums, self._composite_counts = {}, {}
        base = KDConfig(teacher_path=teacher_path, layer_indices=self.composite_config.layer_indices,
                        layer_weights=self.composite_config.layer_weights, pool_size=self.composite_config.pool_size,
                        kd_weight=1.0, warmup_epochs=self.composite_config.gram_warmup_epochs)
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks, kd_config=base)

    @property
    def uses_response(self):
        return "response" in self.composite_config.component_weights

    @property
    def adversarial_names(self):
        return [x for x in self.composite_config.component_weights if x.startswith("adv_")]

    def _setup_train(self):
        super()._setup_train()
        reg_max = int(unwrap_model(self.model).model[-1].reg_max)
        self.composite_loss = CompositeKDLoss(self.composite_config, reg_max=reg_max).to(self.device)
        self.relation_loss = self.composite_loss
        if self.uses_response:
            self.teacher_head_hook = DetectOutputHook(self.teacher_model, detach=True, name="teacher_head")
            self.student_head_hook = DetectOutputHook(unwrap_model(self.model), detach=False, name="student_head")
        if self.adversarial_names:
            specs = {}
            if "adv_p4" in self.adversarial_names: specs["adv_p4"] = (96, True)
            if "adv_p5" in self.adversarial_names: specs["adv_p5"] = (256, False)
            self.adversarial = AdversarialFeatureKD(specs, self.device)
            optimizer_ids = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
            discriminator_ids = {id(p) for d in self.adversarial.discriminators.values() for p in d.parameters()}
            if optimizer_ids & discriminator_ids: raise AssertionError("discriminator entered Student optimizer")
        self.kd_isolation_report.update({
            "kd_method": "composite_exploratory", "composite_config": asdict(self.composite_config),
            "discriminator_absent_from_student_optimizer": True,
            "discriminator_absent_from_student_state_dict": not any("discriminator" in k.lower() for k in unwrap_model(self.model).state_dict()),
        })

    def clear_step_features(self):
        super().clear_step_features()
        for hook in (self.teacher_head_hook, self.student_head_hook):
            if hook is not None: hook.clear()

    def _enable_step_features(self):
        super()._enable_step_features()
        for hook in (self.teacher_head_hook, self.student_head_hook):
            if hook is not None: hook.enable()

    @staticmethod
    def _feature_grad_norm(loss, features):
        grads = torch.autograd.grad(loss, features, retain_graph=True, allow_unused=True)
        return math.sqrt(sum(float(g.detach().float().square().sum()) for g in grads if g is not None))

    def compute_kd_step(self, batch: dict[str, Any], *, epoch: int, batch_index: int, number_of_batches: int):
        self._enable_step_features(); self.teacher_model.eval()
        teacher_start = torch.cuda.Event(enable_timing=True); teacher_end = torch.cuda.Event(enable_timing=True)
        kd_start = torch.cuda.Event(enable_timing=True); kd_end = torch.cuda.Event(enable_timing=True)
        with autocast(self.amp):
            teacher_start.record()
            with torch.no_grad(): self.teacher_model(batch["img"])
            teacher_end.record(); loss_vec, loss_items = self.model(batch); det_loss = loss_vec.sum()
        self.teacher_hooks.validate_once(); self.student_hooks.validate_once()
        tf, sf = self.teacher_hooks.get(), self.student_hooks.get()
        th = self.teacher_head_hook.get() if self.teacher_head_hook else None
        sh = self.student_head_hook.get() if self.student_head_hook else None
        kd_start.record(); raw, details = self.composite_loss.raw_components(sf, tf, batch, sh, th)
        gan_stats = {}
        if self.adversarial is not None:
            s_adv, t_adv = {}, {}
            if "adv_p4" in self.adversarial_names: s_adv["adv_p4"], t_adv["adv_p4"] = sf[18], tf[18]
            if "adv_p5" in self.adversarial_names: s_adv["adv_p5"], t_adv["adv_p5"] = sf[21], tf[21]
            gan_stats = self.adversarial.discriminator_step(s_adv, t_adv)
            raw.update(self.adversarial.generator_losses(s_adv))
        progress = float(epoch) + (batch_index + 1.0) / number_of_batches
        gram_ramp = min(1.0, progress / max(self.composite_config.gram_warmup_epochs, 1e-12))
        adv_ramp = min(1.0, progress / max(self.composite_config.adversarial_warmup_epochs, 1e-12))
        batch_size = int(batch["img"].shape[0]); scaled_parts = {}
        for name, value in raw.items():
            ramp = adv_ramp if name.startswith("adv_") else gram_ramp
            scaled_parts[name] = value * self.composite_config.component_weights[name] * ramp * batch_size
        kd_scaled = torch.stack(list(scaled_parts.values())).sum(); total_loss = det_loss + kd_scaled
        if self.adversarial is not None and batch_index == 0:
            adv_features = [sf[18] for n in self.adversarial_names if n == "adv_p4"] + [sf[21] for n in self.adversarial_names if n == "adv_p5"]
            adv_term = torch.stack([scaled_parts[n] for n in self.adversarial_names]).sum()
            gan_stats["student_adv_grad_norm"] = self._feature_grad_norm(adv_term, adv_features)
            det_grad = self._feature_grad_norm(det_loss, adv_features)
            gan_stats["adv_to_det_grad_ratio"] = gan_stats["student_adv_grad_norm"] / max(det_grad, 1e-12)
        else:
            gan_stats.setdefault("student_adv_grad_norm", 0.0); gan_stats.setdefault("adv_to_det_grad_ratio", 0.0)
        kd_end.record(); kd_end.synchronize()
        if not torch.isfinite(total_loss): raise FloatingPointError("composite total loss is non-finite")
        step = {"loss_vec": loss_vec, "loss_items": loss_items, "det_loss": det_loss,
                "kd_raw": torch.stack(list(raw.values())).sum(), "kd_scaled": kd_scaled,
                "kd_parts": {}, "raw_components": raw, "scaled_components": scaled_parts,
                "component_details": details, "kd_ramp": gram_ramp, "adv_ramp": adv_ramp,
                "total_loss": total_loss, "batch_size": batch_size,
                "teacher_forward_ms": teacher_start.elapsed_time(teacher_end), "relation_loss_ms": kd_start.elapsed_time(kd_end),
                "teacher_hook_counts": dict(self.teacher_hooks.counts), "student_hook_counts": dict(self.student_hooks.counts),
                "teacher_eval": not self.teacher_model.training, "gan_stats": gan_stats}
        self.teacher_eval_always &= step["teacher_eval"]
        return step

    @staticmethod
    def _metric_fields():
        base = ["epoch", "experiment", "gram_ramp_end", "adversarial_ramp_end", "det_loss_mean", "kd_scaled_mean", "kd_to_det_ratio", "teacher_forward_ms", "kd_compute_ms", "peak_cuda_memory_mb"]
        components = ["gram", "mtrd", "foreground", "foreground_cb", "background", "attention", "response", "adv_p4", "adv_p5"]
        relation_details = [f"{component}_{level}" for component in ("gram", "foreground", "foreground_cb", "background") for level in ("p3", "p4", "p5")]
        mtrd_details = [f"mtrd_{level}_{suffix}" for level in ("p3", "p4", "p5") for suffix in ("step1", "step2", "total")]
        mtrd_details += [f"mtrd_transition_{suffix}_{level}" for level in ("p3", "p4", "p5") for suffix in ("entropy", "max_prob")]
        details = relation_details + mtrd_details + ["attention_p3", "attention_p4", "attention_p5", "response_classification", "response_localization"]
        gan = ["d_loss", "g_adv_loss", "d_real_score", "d_fake_score", "d_real_accuracy", "d_fake_accuracy", "d_grad_norm", "student_adv_grad_norm", "adv_to_det_grad_ratio", "gram_to_det_ratio", "discriminator_peak_memory", "discriminator_update_ms"]
        return base + [f"{x}_raw" for x in components] + [f"{x}_scaled" for x in components] + details + gan

    def _prepare_kd_metrics_file(self):
        Path(self.save_dir).mkdir(parents=True, exist_ok=True); self.kd_metrics_path = Path(self.save_dir) / "composite_kd_metrics.csv"
        if not self.args.resume:
            with self.kd_metrics_path.open("w", newline="", encoding="utf-8") as f: csv.DictWriter(f, fieldnames=self._metric_fields()).writeheader()

    def _reset_kd_epoch_stats(self):
        self._kd_epoch_count = 0; self._composite_sums, self._composite_counts = {}, {}; self._last_ramps = (0.0, 0.0)
        if self.device.type == "cuda": torch.cuda.reset_peak_memory_stats(self.device)

    def _add(self, key, value):
        self._composite_sums[key] = self._composite_sums.get(key, 0.0) + float(value)
        self._composite_counts[key] = self._composite_counts.get(key, 0) + 1

    def _update_kd_epoch_stats(self, step):
        self._kd_epoch_count += 1; det = float(step["det_loss"].detach()); scaled = float(step["kd_scaled"].detach())
        self._add("det_loss_mean", det); self._add("kd_scaled_mean", scaled); self._add("kd_to_det_ratio", scaled / max(abs(det), 1e-12))
        self._add("teacher_forward_ms", step["teacher_forward_ms"]); self._add("kd_compute_ms", step["relation_loss_ms"])
        for name, value in step["raw_components"].items(): self._add(f"{name}_raw", value.detach())
        for name, value in step["scaled_components"].items(): self._add(f"{name}_scaled", value.detach())
        for component, values in step["component_details"].items():
            for name, value in values.items(): self._add(f"{component}_{name}", value.detach())
        gan = step["gan_stats"]
        if gan:
            for key, value in gan.items():
                if key not in {"student_adv_grad_norm", "adv_to_det_grad_ratio"} or float(value) != 0.0: self._add(key, value)
            self._add("g_adv_loss", sum(float(step["raw_components"][n].detach()) for n in self.adversarial_names))
        gram_scaled = float(step["scaled_components"].get("gram", torch.tensor(0.0)).detach())
        self._add("gram_to_det_ratio", gram_scaled / max(abs(det), 1e-12)); self._last_ramps = (step["kd_ramp"], step["adv_ramp"])

    def _write_kd_epoch_metrics(self, epoch):
        if not self._kd_epoch_count: return
        n = float(self._kd_epoch_count); row = {key: 0.0 for key in self._metric_fields()}
        row.update({"epoch": epoch + 1, "experiment": self.composite_config.experiment,
                    "gram_ramp_end": self._last_ramps[0], "adversarial_ramp_end": self._last_ramps[1]})
        for key, value in self._composite_sums.items(): row[key] = value / max(self._composite_counts.get(key, int(n)), 1)
        peak = torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == "cuda" else 0.0
        row["peak_cuda_memory_mb"] = peak; row["discriminator_peak_memory"] = peak if self.adversarial else 0.0
        self.kd_peak_cuda_memory_mb = max(self.kd_peak_cuda_memory_mb, peak)
        with self.kd_metrics_path.open("a", newline="", encoding="utf-8") as f: csv.DictWriter(f, fieldnames=self._metric_fields()).writerow(row)
        self.kd_epoch_records.append(row)
        if self.adversarial:
            torch.save(self.adversarial.state_dict(), Path(self.save_dir) / "discriminator_last.pt")
        LOGGER.info(f"CompositeKD: ratio={row['kd_to_det_ratio']:.4f} gram={row['gram_raw']:.5g} peak={peak:.1f}MB")

    def close_kd(self):
        for hook in (self.teacher_head_hook, self.student_head_hook):
            if hook is not None: hook.remove()
        self.teacher_head_hook = self.student_head_hook = None
        super().close_kd()
