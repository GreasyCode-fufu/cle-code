"""Spatially routed mixture-of-experts blocks for dense visual features."""

import copy

import torch
import torch.nn as nn


def _conv_bn_act(c1, c2, k=3, d=1, groups=1):
    """Same-resolution convolution followed by BatchNorm and SiLU."""
    p = ((k - 1) // 2) * d
    return nn.Sequential(
        nn.Conv2d(c1, c2, k, 1, p, dilation=d, groups=groups, bias=False),
        nn.BatchNorm2d(c2),
        nn.SiLU(inplace=True),
    )


class _LocalExpert(nn.Module):
    """Fine-detail expert with a 3x3 depthwise convolution."""

    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(_conv_bn_act(c, c, 3, groups=c), _conv_bn_act(c, c, 1))

    def forward(self, x):
        return self.net(x)


class _DilatedExpert(nn.Module):
    """Context expert for partially occluded regions."""

    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(_conv_bn_act(c, c, 3, d=2, groups=c), _conv_bn_act(c, c, 1))

    def forward(self, x):
        return self.net(x)


class _LargeKernelExpert(nn.Module):
    """Larger receptive-field expert, implemented without extra dependencies."""

    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(_conv_bn_act(c, c, 5, groups=c), _conv_bn_act(c, c, 1))

    def forward(self, x):
        return self.net(x)


class SpatialMoEContext(nn.Module):
    """Per-location MoE context refinement for dense detection features.

    The router predicts expert weights at every spatial location rather than
    globally pooling the image. This lets occluded regions and clear regions
    select different transformations. The module preserves the input channel
    count and exposes ``aux_loss`` for the detection criterion.
    """

    def __init__(self, c1, num_experts=3, top_k=2, reduction=4, balance_weight=0.01):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        if num_experts != 3:
            raise ValueError("SpatialMoEContext currently defines exactly 3 experts")

        hidden = max(c1 // reduction, 16)
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_weight = balance_weight
        self.router = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, num_experts, 1, bias=True),
        )
        self.experts = nn.ModuleList((_LocalExpert(c1), _DilatedExpert(c1), _LargeKernelExpert(c1)))
        self.gamma = nn.Parameter(torch.full((1, c1, 1, 1), 1e-4))
        self.aux_loss = None

    def _topk_weights(self, logits):
        weights = logits.softmax(dim=1)
        if self.top_k < self.num_experts:
            _, indices = weights.topk(self.top_k, dim=1)
            mask = torch.zeros_like(weights).scatter_(1, indices, 1.0)
            weights = weights * mask
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        return weights

    def forward(self, x):
        weights = self._topk_weights(self.router(x))  # [B, E, H, W]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, E, C, H, W]
        mixture = (expert_outputs * weights.unsqueeze(2)).sum(dim=1)

        # KL(mean routing distribution || uniform); zero means balanced usage.
        mean_usage = weights.mean(dim=(0, 2, 3)).clamp_min(1e-6)
        self.aux_loss = self.balance_weight * (mean_usage * (mean_usage * self.num_experts).log()).sum()
        return x + self.gamma * mixture

    def __deepcopy__(self, memo):
        """Exclude the transient, graph-attached auxiliary loss from EMA copies."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, None if key == 'aux_loss' else copy.deepcopy(value, memo))
        return result
