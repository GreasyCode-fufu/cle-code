"""Parameter-free channel-relation knowledge distillation loss."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class ChannelRelationKDLoss(nn.Module):
    """Match per-sample channel Gram matrices at P3/P4/P5."""

    def __init__(
        self,
        layer_indices: tuple[int, ...] = (15, 18, 21),
        layer_weights: tuple[float, ...] = (1.0, 1.0, 1.0),
        pool_size: int = 20,
        eps: float = 1e-6,
        center: bool = True,
    ) -> None:
        super().__init__()
        self.layer_indices = tuple(int(i) for i in layer_indices)
        self.layer_weights = tuple(float(w) for w in layer_weights)
        self.pool_size = int(pool_size)
        self.eps = float(eps)
        self.center = bool(center)
        if not self.layer_indices or len(self.layer_indices) != len(self.layer_weights):
            raise ValueError("layer_indices and layer_weights must be non-empty and have equal length")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("layer_indices must be unique")
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if any(w < 0 for w in self.layer_weights) or sum(self.layer_weights) <= 0:
            raise ValueError("layer_weights must be non-negative with a positive sum")

    def _ordered(self, features: Mapping[int, torch.Tensor] | Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(features, Mapping):
            missing = [i for i in self.layer_indices if i not in features]
            if missing:
                raise KeyError(f"missing relation features for layers {missing}")
            return [features[i] for i in self.layer_indices]
        values = list(features)
        if len(values) != len(self.layer_indices):
            raise ValueError(f"expected {len(self.layer_indices)} features, got {len(values)}")
        return values

    def _gram(self, feature: torch.Tensor, *, teacher: bool) -> torch.Tensor:
        if not isinstance(feature, torch.Tensor):
            raise TypeError(f"feature must be Tensor, got {type(feature).__name__}")
        if feature.ndim != 4:
            raise ValueError(f"feature must have BCHW shape, got {tuple(feature.shape)}")
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
        gram = torch.bmm(x, x.transpose(1, 2))
        if not torch.isfinite(gram).all():
            raise FloatingPointError("channel Gram matrix contains NaN or Inf")
        return gram.detach() if teacher else gram

    def forward(
        self,
        student_features: Mapping[int, torch.Tensor] | Sequence[torch.Tensor],
        teacher_features: Mapping[int, torch.Tensor] | Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        student = self._ordered(student_features)
        teacher = self._ordered(teacher_features)
        names = ("p3", "p4", "p5")
        parts: dict[str, torch.Tensor] = {}
        weighted: list[torch.Tensor] = []

        # Explicitly disable AMP: all Gram math is float32 even when hook inputs are float16.
        device_type = student[0].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            for position, (index, weight, feat_s, feat_t) in enumerate(
                zip(self.layer_indices, self.layer_weights, student, teacher)
            ):
                if tuple(feat_s.shape) != tuple(feat_t.shape):
                    raise ValueError(
                        f"layer {index} Student/Teacher shape mismatch: "
                        f"{tuple(feat_s.shape)} != {tuple(feat_t.shape)}"
                    )
                gram_s = self._gram(feat_s, teacher=False)
                gram_t = self._gram(feat_t, teacher=True)
                channels = feat_s.shape[1]
                diff = gram_s - gram_t
                loss_scale = (diff.square().sum(dim=(1, 2)) / channels).mean()
                if not torch.isfinite(loss_scale):
                    raise FloatingPointError(f"layer {index} relation loss contains NaN or Inf")
                name = names[position] if position < len(names) else f"layer_{index}"
                parts[name] = loss_scale
                weighted.append(loss_scale * weight)
            total = torch.stack(weighted).sum() / sum(self.layer_weights)
        if total.dtype != torch.float32:
            raise RuntimeError(f"relation loss must be float32, got {total.dtype}")
        return total, parts
