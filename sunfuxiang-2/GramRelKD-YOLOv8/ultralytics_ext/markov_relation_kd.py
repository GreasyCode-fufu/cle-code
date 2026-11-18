"""Parameter-free multi-step transition relation distillation for channel features."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class MultiStepTransitionKDLoss(nn.Module):
    """Match Teacher-to-Student channel transition distributions at multiple steps."""

    def __init__(
        self,
        layer_indices: tuple[int, ...] = (15, 18, 21),
        layer_weights: tuple[float, ...] = (1.0, 1.0, 1.0),
        pool_size: int = 20,
        temperature: float = 0.2,
        transition_steps: tuple[int, ...] = (1, 2),
        step_weights: tuple[float, ...] = (1.0, 1.0),
        self_loop_prob: float = 0.0,
        divergence: str = "kl",
        center: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.layer_indices = tuple(int(i) for i in layer_indices)
        self.layer_weights = tuple(float(w) for w in layer_weights)
        self.pool_size = int(pool_size)
        self.temperature = float(temperature)
        self.transition_steps = tuple(int(step) for step in transition_steps)
        self.step_weights = tuple(float(w) for w in step_weights)
        self.self_loop_prob = float(self_loop_prob)
        self.divergence = str(divergence).lower()
        self.center = bool(center)
        self.eps = float(eps)
        self._validate_config()

    def _validate_config(self) -> None:
        if not self.layer_indices or len(self.layer_indices) != len(self.layer_weights):
            raise ValueError("layer_indices and layer_weights must be non-empty and have equal length")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("layer_indices must be unique")
        if any(w < 0 for w in self.layer_weights) or sum(self.layer_weights) <= 0:
            raise ValueError("layer_weights must be non-negative with a positive sum")
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not self.transition_steps:
            raise ValueError("transition_steps must be non-empty")
        if any(step <= 0 for step in self.transition_steps):
            raise ValueError("transition_steps must contain positive integers")
        if tuple(sorted(set(self.transition_steps))) != self.transition_steps:
            raise ValueError("transition_steps must be strictly increasing and unique")
        if 1 not in self.transition_steps:
            raise ValueError("transition_steps must include step 1")
        if len(self.step_weights) != len(self.transition_steps):
            raise ValueError("step_weights length must match transition_steps")
        if any(w < 0 for w in self.step_weights) or sum(self.step_weights) <= 0:
            raise ValueError("step_weights must be non-negative with a positive sum")
        if not 0.0 <= self.self_loop_prob < 1.0:
            raise ValueError("self_loop_prob must be in [0, 1)")
        if self.divergence != "kl":
            raise ValueError("only divergence='kl' is supported")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def _ordered(self, features: Mapping[int, torch.Tensor] | Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(features, Mapping):
            missing = [index for index in self.layer_indices if index not in features]
            if missing:
                raise KeyError(f"missing transition features for layers {missing}")
            return [features[index] for index in self.layer_indices]
        values = list(features)
        if len(values) != len(self.layer_indices):
            raise ValueError(f"expected {len(self.layer_indices)} features, got {len(values)}")
        return values

    def _preprocess(self, feature: torch.Tensor, *, teacher: bool) -> torch.Tensor:
        if not isinstance(feature, torch.Tensor):
            raise TypeError(f"feature must be Tensor, got {type(feature).__name__}")
        if feature.ndim != 4:
            raise ValueError(f"feature must have BCHW shape, got {tuple(feature.shape)}")
        if feature.shape[1] < 2:
            raise ValueError(f"MTRD requires at least 2 channels, got C={feature.shape[1]}")
        if not torch.isfinite(feature).all():
            raise FloatingPointError("feature contains NaN or Inf")
        x = feature.detach() if teacher else feature
        x = x.float()
        if tuple(x.shape[-2:]) != (self.pool_size, self.pool_size):
            x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        x = x.flatten(2)
        if self.center:
            x = x - x.mean(dim=-1, keepdim=True)
        x = F.normalize(x, p=2, dim=-1, eps=self.eps)
        if not torch.isfinite(x).all():
            raise FloatingPointError("preprocessed feature contains NaN or Inf")
        return x.detach() if teacher else x

    def transition_matrices(
        self, feature: torch.Tensor, *, teacher: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return non-diagonal Q and fixed-self-loop P, both shaped [B, C, C]."""
        device_type = feature.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            x = self._preprocess(feature, teacher=teacher)
            similarity = torch.bmm(x, x.transpose(1, 2))
            channels = similarity.shape[-1]
            diagonal = torch.eye(channels, dtype=torch.bool, device=similarity.device).unsqueeze(0)
            logits = similarity.masked_fill(diagonal, float("-inf")) / self.temperature
            q = torch.softmax(logits, dim=-1)
            identity = torch.eye(channels, dtype=q.dtype, device=q.device).unsqueeze(0)
            p = self.self_loop_prob * identity + (1.0 - self.self_loop_prob) * q
            if not torch.isfinite(q).all() or not torch.isfinite(p).all():
                raise FloatingPointError("transition matrix contains NaN or Inf")
            if torch.any(q < 0) or torch.any(p < 0):
                raise FloatingPointError("transition matrix contains negative probabilities")
            tolerance = 5e-5
            if not torch.allclose(q.sum(dim=-1), torch.ones_like(q.sum(dim=-1)), atol=tolerance, rtol=0):
                raise FloatingPointError("Q transition rows do not sum to 1")
            if not torch.allclose(p.sum(dim=-1), torch.ones_like(p.sum(dim=-1)), atol=tolerance, rtol=0):
                raise FloatingPointError("P transition rows do not sum to 1")
        if teacher:
            return q.detach(), p.detach()
        return q, p

    def transition_powers(self, p: torch.Tensor) -> dict[int, torch.Tensor]:
        """Build requested positive powers with bmm and row renormalization."""
        if p.ndim != 3 or p.shape[-1] != p.shape[-2]:
            raise ValueError(f"P must have [B,C,C] shape, got {tuple(p.shape)}")
        powers: dict[int, torch.Tensor] = {1: p}
        current = p
        for step in range(2, max(self.transition_steps) + 1):
            current = torch.bmm(current, p)
            current = current.clamp(min=self.eps)
            current = current / current.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            if not torch.isfinite(current).all():
                raise FloatingPointError(f"transition power P^{step} contains NaN or Inf")
            if step in self.transition_steps:
                powers[step] = current
        return {step: powers[step] for step in self.transition_steps}

    def transition_diagnostics(self, feature: torch.Tensor, *, teacher: bool = True) -> dict[str, torch.Tensor]:
        """Return per-row entropy and concentration diagnostics without changing the loss."""
        q, p = self.transition_matrices(feature, teacher=teacher)
        channels = q.shape[-1]
        entropy = -(q * torch.log(q.clamp(min=self.eps))).sum(dim=-1)
        if channels == 2:
            normalized_entropy = torch.zeros_like(entropy)
        else:
            normalized_entropy = entropy / math.log(channels - 1)
        max_prob = q.max(dim=-1).values
        topk = min(5, channels - 1)
        top5_mass = q.topk(k=topk, dim=-1).values.sum(dim=-1)
        p2 = torch.bmm(p, p).clamp(min=self.eps)
        p2 = p2 / p2.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        entropy2 = -(p2 * torch.log(p2.clamp(min=self.eps))).sum(dim=-1)
        normalized_entropy2 = entropy2 / math.log(channels) if channels > 1 else torch.zeros_like(entropy2)
        diagnostics = {
            "normalized_entropy_rows": normalized_entropy,
            "max_probability_rows": max_prob,
            "top5_mass_rows": top5_mass,
            "step2_normalized_entropy_rows": normalized_entropy2,
        }
        if any(not torch.isfinite(value).all() for value in diagnostics.values()):
            raise FloatingPointError("transition diagnostics contain NaN or Inf")
        return {key: value.detach() if teacher else value for key, value in diagnostics.items()}

    def _row_kl(self, teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
        teacher = teacher.detach()
        value = (
            teacher
            * (torch.log(teacher.clamp(min=self.eps)) - torch.log(student.clamp(min=self.eps)))
        ).sum(dim=-1).mean()
        if not torch.isfinite(value):
            raise FloatingPointError("MTRD KL loss contains NaN or Inf")
        return value.float()

    def forward(
        self,
        student_features: Mapping[int, torch.Tensor] | Sequence[torch.Tensor],
        teacher_features: Mapping[int, torch.Tensor] | Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        student = self._ordered(student_features)
        teacher = self._ordered(teacher_features)
        names = ("p3", "p4", "p5")
        parts: dict[str, torch.Tensor] = {}
        layer_totals: list[torch.Tensor] = []

        device_type = student[0].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            for position, (index, layer_weight, feat_s, feat_t) in enumerate(
                zip(self.layer_indices, self.layer_weights, student, teacher)
            ):
                if tuple(feat_s.shape) != tuple(feat_t.shape):
                    raise ValueError(
                        f"layer {index} Student/Teacher shape mismatch: "
                        f"{tuple(feat_s.shape)} != {tuple(feat_t.shape)}"
                    )
                name = names[position] if position < len(names) else f"layer_{index}"
                q_s, p_s = self.transition_matrices(feat_s, teacher=False)
                q_t, p_t = self.transition_matrices(feat_t, teacher=True)
                powers_s = self.transition_powers(p_s)
                powers_t = {step: value.detach() for step, value in self.transition_powers(p_t).items()}
                step_losses: list[torch.Tensor] = []
                for step, step_weight in zip(self.transition_steps, self.step_weights):
                    step_loss = self._row_kl(powers_t[step], powers_s[step])
                    parts[f"{name}_step{step}"] = step_loss
                    step_losses.append(step_loss * step_weight)
                layer_total = torch.stack(step_losses).sum() / sum(self.step_weights)
                parts[f"{name}_total"] = layer_total
                layer_totals.append(layer_total * layer_weight)

                channels = q_t.shape[-1]
                entropy = -(q_t * torch.log(q_t.clamp(min=self.eps))).sum(dim=-1)
                normalized_entropy = (
                    torch.zeros_like(entropy)
                    if channels == 2
                    else entropy / math.log(channels - 1)
                )
                parts[f"transition_entropy_{name}"] = normalized_entropy.mean().float()
                parts[f"transition_max_prob_{name}"] = q_t.max(dim=-1).values.mean().float()

            total = torch.stack(layer_totals).sum() / sum(self.layer_weights)

        if total.dtype != torch.float32:
            raise RuntimeError(f"MTRD loss must be float32, got {total.dtype}")
        if not torch.isfinite(total):
            raise FloatingPointError("MTRD total loss contains NaN or Inf")
        return total, parts
