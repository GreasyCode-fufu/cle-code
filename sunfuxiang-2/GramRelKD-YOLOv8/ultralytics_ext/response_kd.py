"""Version-locked response and localization distillation for Detect outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DetectOutputHook:
    def __init__(self, model, detach=False, name="head"):
        self.detach, self.name, self.active, self.output, self.count = detach, name, False, None, 0
        self.handle = model.model[-1].register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        if not self.active:
            return
        raw = output[1] if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], dict) else output
        if not isinstance(raw, dict) or not {"boxes", "scores"}.issubset(raw):
            raise TypeError(f"{self.name}: expected Detect training dict")
        self.count += 1
        self.output = {k: (v.detach() if self.detach and torch.is_tensor(v) else v) for k, v in raw.items()}

    def enable(self):
        self.output, self.count, self.active = None, 0, True

    def clear(self):
        self.active, self.output, self.count = False, None, 0

    def get(self):
        if self.count != 1 or self.output is None:
            raise RuntimeError(f"{self.name}: Detect hook count={self.count}")
        return self.output

    def remove(self):
        self.clear(); self.handle.remove()


class ResponseLocalizationKDLoss(nn.Module):
    """Soft class response BCE plus teacher-confidence top-k DFL KL."""

    def __init__(self, reg_max=16, class_fraction=0.4, localization_fraction=0.6, topk=100, eps=1e-6):
        super().__init__()
        self.reg_max, self.class_fraction, self.localization_fraction = int(reg_max), float(class_fraction), float(localization_fraction)
        self.topk, self.eps = int(topk), float(eps)

    def forward(self, student: dict, teacher: dict):
        s_scores, t_scores = student["scores"].float(), teacher["scores"].detach().float()
        t_prob = t_scores.sigmoid()
        confidence = t_prob.max(dim=1, keepdim=True).values
        cls_map = F.binary_cross_entropy_with_logits(s_scores, t_prob, reduction="none")
        cls_loss = (cls_map * confidence).sum() / (confidence.sum() * s_scores.shape[1] + self.eps)

        b, channels, anchors = student["boxes"].shape
        if channels != 4 * self.reg_max or teacher["boxes"].shape != student["boxes"].shape:
            raise ValueError("unexpected Detect box distribution shape")
        k = min(self.topk, anchors)
        chosen = confidence.squeeze(1).topk(k, dim=1).indices
        s_box = student["boxes"].float().view(b, 4, self.reg_max, anchors).permute(0, 1, 3, 2)
        t_box = teacher["boxes"].detach().float().view(b, 4, self.reg_max, anchors).permute(0, 1, 3, 2)
        gather = chosen[:, None, :, None].expand(-1, 4, -1, self.reg_max)
        s_sel, t_sel = s_box.gather(2, gather), t_box.gather(2, gather)
        t_distribution = t_sel.softmax(dim=-1)
        loc_loss = F.kl_div(s_sel.log_softmax(dim=-1), t_distribution, reduction="batchmean") / (4 * k)
        total = self.class_fraction * cls_loss + self.localization_fraction * loc_loss
        if not torch.isfinite(total):
            raise FloatingPointError("response KD is non-finite")
        return total, {"classification": cls_loss, "localization": loc_loss}
