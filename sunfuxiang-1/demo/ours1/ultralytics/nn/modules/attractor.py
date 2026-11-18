# Ultralytics YOLO, AGPL-3.0 license
"""Object Attractor refinement modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class ObjectAttractorRefinement(nn.Module):
    """Lightweight post-head object particle refinement for YOLO detections.

    Input predictions use shape [B, 4 + nc, N]. For the standard YOLOv8 Detect
    head in this repository, the first four channels are decoded xywh boxes.
    Internally this module refines boxes in xywh and returns the same format.
    Only per-image top-k particles are returned to keep the MVP inexpensive.
    """

    def __init__(
        self,
        nc: int,
        enabled: bool = False,
        num_steps: int = 1,
        hidden_dim: int = 128,
        topk: int = 300,
        refine_box: bool = True,
        refine_score: bool = False,
        refine_cls: bool = False,
        delta_scale: float = 0.05,
        use_particle_interaction: bool = False,
        box_format: str = "xywh",
    ):
        super().__init__()
        if box_format not in {"xywh", "xyxy"}:
            raise ValueError(f"Unsupported box_format={box_format!r}. Expected 'xywh' or 'xyxy'.")

        self.nc = nc
        self.enabled = enabled
        self.num_steps = max(int(num_steps), 0)
        self.hidden_dim = int(hidden_dim)
        self.topk = int(topk)
        self.refine_box = refine_box
        self.refine_score = refine_score
        self.refine_cls = refine_cls
        self.delta_scale = float(delta_scale)
        self.use_particle_interaction = use_particle_interaction
        self.box_format = box_format

        self.state_dim = 5 + nc  # normalized box(4) + best score(1) + class probabilities(nc)
        self.delta_dim = 4 + int(refine_score) + (nc if refine_cls else 0)
        self.mlp = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.delta_dim),
        )

        # Start as a no-op before any refinement-specific training.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, preds: torch.Tensor, features: list[torch.Tensor] | None = None) -> torch.Tensor:
        """Refine decoded YOLO predictions before NMS."""
        if not self.enabled or self.num_steps == 0 or self.topk <= 0 or preds.ndim != 3:
            return preds
        if preds.shape[1] < 4 + self.nc:
            return preds
        # TODO: add phase-two particle interaction / attract-repel dynamics here.

        boxes = preds[:, :4].transpose(1, 2).contiguous()
        cls = preds[:, 4 : 4 + self.nc].transpose(1, 2).contiguous()
        extra = preds[:, 4 + self.nc :].transpose(1, 2).contiguous()

        boxes_xywh = self._to_xywh(boxes)
        scores = cls.amax(dim=-1)
        k = min(self.topk, preds.shape[-1])
        topk_idx = scores.topk(k, dim=1).indices
        boxes_xywh = self._gather_particles(boxes_xywh, topk_idx)
        cls = self._gather_particles(cls, topk_idx)
        extra = self._gather_particles(extra, topk_idx) if extra.shape[-1] else extra[:, :k]

        for _ in range(self.num_steps):
            # TODO: sample multi-scale features at particle centers and concatenate them to the state.
            state = self._particle_state(boxes_xywh, cls)
            delta = self.mlp(state)
            offset = 0

            if self.refine_box:
                delta_box = delta[..., :4].tanh()
                boxes_xywh = self._apply_box_delta(boxes_xywh, delta_box)
            offset += 4

            if self.refine_score:
                delta_score = delta[..., offset : offset + 1].tanh()
                cls = self._apply_score_delta(cls, delta_score)
                offset += 1

            if self.refine_cls:
                delta_cls = delta[..., offset : offset + self.nc].tanh()
                cls = (self._safe_logit(cls) + delta_cls * self.delta_scale).sigmoid()

        refined_boxes = self._from_xywh(boxes_xywh)
        refined = torch.cat((refined_boxes, cls, extra), dim=-1) if extra.shape[-1] else torch.cat((refined_boxes, cls), dim=-1)
        return refined.transpose(1, 2).contiguous()

    def loss(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Placeholder for future refinement-specific auxiliary losses."""
        device = next(self.parameters()).device
        # TODO: add attractor auxiliary losses for refined boxes/classes during training.
        return {"attractor_loss": torch.zeros((), device=device)}

    def _particle_state(self, boxes_xywh: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        img_wh = self._estimate_image_wh(boxes_xywh)
        norm = torch.cat((img_wh, img_wh), dim=-1).clamp_min(1.0)
        box_state = boxes_xywh / norm
        score = cls.amax(dim=-1, keepdim=True)
        return torch.cat((box_state, score, cls), dim=-1)

    def _apply_box_delta(self, boxes_xywh: torch.Tensor, delta_box: torch.Tensor) -> torch.Tensor:
        xy, wh = boxes_xywh.split((2, 2), dim=-1)
        wh = wh.clamp_min(1e-3)
        xy = xy + delta_box[..., :2] * self.delta_scale * wh
        wh = wh * torch.exp(delta_box[..., 2:] * self.delta_scale)
        return self._clamp_xywh(torch.cat((xy, wh), dim=-1))

    def _apply_score_delta(self, cls: torch.Tensor, delta_score: torch.Tensor) -> torch.Tensor:
        score = cls.amax(dim=-1, keepdim=True).clamp(1e-6, 1 - 1e-6)
        refined_score = (self._safe_logit(score) + delta_score * self.delta_scale).sigmoid()
        return (cls * (refined_score / score)).clamp(0, 1)

    def _clamp_xywh(self, boxes_xywh: torch.Tensor) -> torch.Tensor:
        img_wh = self._estimate_image_wh(boxes_xywh)
        xy = torch.minimum(torch.maximum(boxes_xywh[..., :2], torch.zeros_like(boxes_xywh[..., :2])), img_wh)
        wh = torch.minimum(boxes_xywh[..., 2:].clamp_min(1e-3), img_wh.clamp_min(1e-3))
        return torch.cat((xy, wh), dim=-1)

    def _estimate_image_wh(self, boxes_xywh: torch.Tensor) -> torch.Tensor:
        half_wh = boxes_xywh[..., 2:].clamp_min(0) * 0.5
        max_xy = (boxes_xywh[..., :2] + half_wh).amax(dim=1, keepdim=True)
        return max_xy.clamp_min(1.0)

    def _to_xywh(self, boxes: torch.Tensor) -> torch.Tensor:
        if self.box_format == "xywh":
            return boxes
        x1, y1, x2, y2 = boxes.unbind(-1)
        return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1).clamp_min(0), (y2 - y1).clamp_min(0)), dim=-1)

    def _from_xywh(self, boxes_xywh: torch.Tensor) -> torch.Tensor:
        if self.box_format == "xywh":
            return boxes_xywh
        x, y, w, h = boxes_xywh.unbind(-1)
        return torch.stack((x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5), dim=-1)

    @staticmethod
    def _gather_particles(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

    @staticmethod
    def _safe_logit(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(1e-6, 1 - 1e-6)
        return torch.log(x / (1 - x))
