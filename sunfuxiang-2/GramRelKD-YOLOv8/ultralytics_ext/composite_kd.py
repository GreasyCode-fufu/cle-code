"""Composable, parameter-isolated training-time KD objectives."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .attention_kd import SpatialAttentionKDLoss
from .foreground_relation_kd import MaskedChannelRelationKDLoss
from .markov_relation_kd import MultiStepTransitionKDLoss
from .relation_kd import ChannelRelationKDLoss
from .response_kd import ResponseLocalizationKDLoss


@dataclass
class CompositeKDConfig:
    experiment: str
    component_weights: dict[str, float] = field(default_factory=dict)
    class_weights: tuple[float, ...] | None = None
    layer_indices: tuple[int, ...] = (15, 18, 21)
    layer_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    pool_size: int = 20
    gram_warmup_epochs: float = 10.0
    adversarial_warmup_epochs: float = 20.0


class CompositeKDLoss(nn.Module):
    def __init__(self, config: CompositeKDConfig, reg_max=16):
        super().__init__(); self.config = config
        kw = dict(layer_indices=config.layer_indices, layer_weights=config.layer_weights, pool_size=config.pool_size, eps=1e-6, center=True)
        self.gram = ChannelRelationKDLoss(**kw)
        self.mtrd = MultiStepTransitionKDLoss(layer_indices=config.layer_indices, layer_weights=config.layer_weights, pool_size=config.pool_size, temperature=0.05, transition_steps=(1, 2), step_weights=(1.0, 1.0), self_loop_prob=0.0, divergence="kl", center=True, eps=1e-6)
        self.foreground = MaskedChannelRelationKDLoss(mode="foreground", **kw)
        self.foreground_cb = MaskedChannelRelationKDLoss(mode="foreground", class_weights=config.class_weights, **kw)
        self.background = MaskedChannelRelationKDLoss(mode="background", **kw)
        self.attention = SpatialAttentionKDLoss(config.layer_indices, config.layer_weights)
        self.response = ResponseLocalizationKDLoss(reg_max=reg_max)

    def raw_components(self, student, teacher, batch=None, student_head=None, teacher_head=None):
        wanted = self.config.component_weights; raw, details = {}, {}
        if "gram" in wanted: raw["gram"], details["gram"] = self.gram(student, teacher)
        if "mtrd" in wanted: raw["mtrd"], details["mtrd"] = self.mtrd(student, teacher)
        if "foreground" in wanted: raw["foreground"], details["foreground"] = self.foreground(student, teacher, batch)
        if "foreground_cb" in wanted: raw["foreground_cb"], details["foreground_cb"] = self.foreground_cb(student, teacher, batch)
        if "background" in wanted: raw["background"], details["background"] = self.background(student, teacher, batch)
        if "attention" in wanted: raw["attention"], details["attention"] = self.attention(student, teacher)
        if "response" in wanted: raw["response"], details["response"] = self.response(student_head, teacher_head)
        for name, value in raw.items():
            if not torch.isfinite(value): raise FloatingPointError(f"{name} KD is non-finite")
        return raw, details
