"""Soft foreground/background channel-relation losses."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .relation_kd import ChannelRelationKDLoss


def soft_box_masks(batch: dict, features: Mapping[int, torch.Tensor], class_weights=None, taper=0.10, eps=1e-6):
    """Build smooth masks in the already-letterboxed normalized image coordinate system."""
    boxes = batch["bboxes"].float()
    batch_idx = batch["batch_idx"].long().view(-1)
    classes = batch["cls"].long().view(-1)
    result = {}
    for idx, feature in features.items():
        b, _, h, w = feature.shape
        yy = (torch.arange(h, device=feature.device, dtype=torch.float32) + 0.5) / h
        xx = (torch.arange(w, device=feature.device, dtype=torch.float32) + 0.5) / w
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        mask = torch.zeros((b, 1, h, w), device=feature.device, dtype=torch.float32)
        for n in range(boxes.shape[0]):
            bi = int(batch_idx[n])
            if bi < 0 or bi >= b:
                continue
            cx, cy, bw, bh = boxes[n]
            tau = torch.clamp(torch.minimum(bw, bh) * taper, min=1.0 / max(h, w), max=0.15)
            x1, x2, y1, y2 = cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2
            smooth = torch.sigmoid((xx - x1) / tau) * torch.sigmoid((x2 - xx) / tau)
            smooth = smooth * torch.sigmoid((yy - y1) / tau) * torch.sigmoid((y2 - yy) / tau)
            if class_weights is not None:
                smooth = smooth * class_weights[classes[n]].float()
            mask[bi, 0] = torch.maximum(mask[bi, 0], smooth)
        result[idx] = mask.clamp(min=eps, max=2.0 if class_weights is not None else 1.0)
    return result


class MaskedChannelRelationKDLoss(nn.Module):
    """Apply a shared soft mask before the verified channel Gram objective."""

    def __init__(self, mode="foreground", class_weights=None, **gram_kwargs):
        super().__init__()
        if mode not in {"foreground", "background"}:
            raise ValueError("mode must be foreground or background")
        self.mode = mode
        self.gram = ChannelRelationKDLoss(**gram_kwargs)
        weights = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else torch.empty(0)
        self.register_buffer("class_weights", weights, persistent=False)

    def forward(self, student, teacher, batch):
        if batch["bboxes"].numel() == 0 and self.mode == "foreground":
            zero = sum(x.float().sum() for x in student.values()) * 0.0
            return zero, {"p3": zero, "p4": zero, "p5": zero}
        weights = self.class_weights if self.class_weights.numel() else None
        masks = soft_box_masks(batch, student, weights)
        masked_s, masked_t = {}, {}
        for idx in student:
            mask = masks[idx]
            if self.mode == "background":
                mask = (1.0 - mask.clamp(max=1.0)).clamp(1e-6, 1.0)
            scale = mask.sqrt()
            masked_s[idx] = student[idx].float() * scale
            masked_t[idx] = teacher[idx].detach().float() * scale
        return self.gram(masked_s, masked_t)
