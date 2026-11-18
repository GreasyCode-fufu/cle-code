"""Training-only spectral-normalized discriminators for feature KD."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class FeatureDiscriminator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, 128, 1)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 64, 3, 2, 1)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 32, 3, 2, 1)), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), spectral_norm(nn.Linear(32, 1)),
        )

    def forward(self, x):
        return self.net(x)


def normalize_feature(x: torch.Tensor, pool20=False, eps=1e-6):
    x = x.float()
    if pool20 and tuple(x.shape[-2:]) != (20, 20):
        x = F.adaptive_avg_pool2d(x, (20, 20))
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    return F.normalize(x.flatten(2), p=2, dim=2, eps=eps).view_as(x)


class AdversarialFeatureKD:
    """Owns independent discriminators and their optimizers, never the Student model."""

    def __init__(self, specs: dict[str, tuple[int, bool]], device):
        self.discriminators = {name: FeatureDiscriminator(ch).to(device) for name, (ch, _) in specs.items()}
        self.optimizers = {name: torch.optim.AdamW(d.parameters(), lr=1e-4, betas=(0.0, 0.9), weight_decay=0.0) for name, d in self.discriminators.items()}
        self.specs, self.device = specs, device

    @staticmethod
    def _grad_norm(module):
        return sum((p.grad.detach().float().norm().item() ** 2 for p in module.parameters() if p.grad is not None), 0.0) ** 0.5

    def discriminator_step(self, student: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor]):
        start = time.perf_counter(); totals = {"d_loss": 0.0, "d_real_score": 0.0, "d_fake_score": 0.0, "d_real_accuracy": 0.0, "d_fake_accuracy": 0.0, "d_grad_norm": 0.0}
        for name, discriminator in self.discriminators.items():
            optimizer = self.optimizers[name]; optimizer.zero_grad(set_to_none=True)
            _, pool20 = self.specs[name]
            real = normalize_feature(teacher[name].detach(), pool20)
            fake = normalize_feature(student[name].detach(), pool20)
            real_score, fake_score = discriminator(real), discriminator(fake)
            loss = F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
            if not torch.isfinite(loss): raise FloatingPointError("discriminator loss is non-finite")
            loss.backward(); optimizer.step()
            totals["d_loss"] += float(loss.detach()); totals["d_real_score"] += float(real_score.mean().detach())
            totals["d_fake_score"] += float(fake_score.mean().detach()); totals["d_real_accuracy"] += float((real_score > 0).float().mean())
            totals["d_fake_accuracy"] += float((fake_score < 0).float().mean()); totals["d_grad_norm"] += self._grad_norm(discriminator)
        n = len(self.discriminators)
        for key in totals: totals[key] /= n
        totals["discriminator_update_ms"] = (time.perf_counter() - start) * 1000
        return totals

    def generator_losses(self, student: dict[str, torch.Tensor]):
        result = {}
        for name, discriminator in self.discriminators.items():
            for p in discriminator.parameters(): p.requires_grad_(False)
            _, pool20 = self.specs[name]
            result[name] = -discriminator(normalize_feature(student[name], pool20)).mean()
            for p in discriminator.parameters(): p.requires_grad_(True)
        return result

    def state_dict(self):
        return {"discriminators": {k: v.state_dict() for k, v in self.discriminators.items()}, "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()}}
