"""Parameter-free spatial attention transfer for detector feature maps."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn


class SpatialAttentionKDLoss(nn.Module):
    """Match per-sample L2-normalized spatial energy maps."""

    def __init__(self, layer_indices=(15, 18, 21), layer_weights=(1.0, 1.0, 1.0), eps=1e-6):
        super().__init__()
        self.layer_indices = tuple(int(x) for x in layer_indices)
        self.layer_weights = tuple(float(x) for x in layer_weights)
        self.eps = float(eps)
        if len(self.layer_indices) != len(self.layer_weights) or sum(self.layer_weights) <= 0:
            raise ValueError("invalid attention layer configuration")

    def _attention(self, feature: torch.Tensor, detach: bool) -> torch.Tensor:
        x = feature.detach() if detach else feature
        x = x.float().square().sum(dim=1).flatten(1)
        return F.normalize(x, p=2, dim=1, eps=self.eps)

    def forward(self, student: Mapping[int, torch.Tensor], teacher: Mapping[int, torch.Tensor]):
        parts, weighted = {}, []
        names = ("p3", "p4", "p5")
        with torch.autocast(device_type=next(iter(student.values())).device.type, enabled=False):
            for pos, (idx, weight) in enumerate(zip(self.layer_indices, self.layer_weights)):
                if student[idx].shape != teacher[idx].shape:
                    raise ValueError(f"attention feature mismatch at layer {idx}")
                value = F.mse_loss(self._attention(student[idx], False), self._attention(teacher[idx], True))
                if not torch.isfinite(value):
                    raise FloatingPointError("attention loss is non-finite")
                name = names[pos] if pos < 3 else f"layer_{idx}"
                parts[name] = value
                weighted.append(value * weight)
        return torch.stack(weighted).sum() / sum(self.layer_weights), parts
