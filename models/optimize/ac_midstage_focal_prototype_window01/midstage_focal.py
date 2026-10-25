"""Midstage focal contrast: an unconfirmed adaptation of mature FocalNet operators.

Microsoft FocalNet (NeurIPS 2022), MIT attribution and original source are retained
in references. This is neither a full FocalNet reproduction nor a novelty claim.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .c4_moment_stem import C4MomentStem, counted_flops as moment_flops

ARMS = ['mom_focalcontrast', 'focalcontrast_only', 'mom_focalraw',
        'mom_misaligned', 'mom_additive']


class FocalCore(nn.Module):
    def __init__(self, rank=24, kind='contrast', spatial=(28, 28), padding_mode='replicate'):
        super().__init__()
        assert kind in ['contrast', 'raw', 'misaligned', 'additive']
        self.rank, self.kind = rank, kind
        # Centering cancels a constant modulator bias and the global gate bias.
        # Both projections are biasless in every arm to avoid dead parameters.
        self.f = nn.Conv2d(rank, 2 * rank + 3, 1, bias=False)
        self.h = nn.Conv2d(rank, rank, 1, bias=False)
        self.layers = nn.ModuleList([
            nn.Conv2d(rank, rank, k, padding=k//2, groups=rank,
                      bias=False, padding_mode=padding_mode) for k in [3, 5]])
        if kind == 'misaligned':
            self.register_buffer('permutation', torch.tensor(
                np.random.default_rng(20261025).permutation(spatial[0] * spatial[1]), dtype=torch.long))

    def parts(self, x):
        query, context, gates = self.f(x).split([self.rank, self.rank, 3], 1)
        total = 0
        for index, layer in enumerate(self.layers):
            context = F.gelu(layer(context))
            total = total + context * gates[:, index:index+1]
        global_context = F.gelu(context.mean(2, keepdim=True).mean(3, keepdim=True))
        total = total + global_context * gates[:, 2:3]
        return query, self.h(total)

    def forward(self, x):
        query, modulator = self.parts(x)
        if self.kind != 'raw':
            modulator = modulator - modulator.mean((2, 3), keepdim=True)
        if self.kind == 'misaligned':
            assert x.shape[2] * x.shape[3] == self.permutation.numel()
            modulator = modulator.flatten(2)[:, :, self.permutation].reshape_as(modulator)
        return query + modulator if self.kind == 'additive' else query * modulator


class FocalResidual(nn.Module):
    def __init__(self, kind, channels=192, rank=24, spatial=(28, 28)):
        super().__init__()
        self.channels, self.rank = channels, rank
        self.down = nn.Conv2d(channels, rank, 1, bias=False)
        self.core = FocalCore(rank, kind, spatial)
        self.up = nn.Conv2d(rank, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def descriptor(self, x):
        assert x.ndim == 4 and x.shape[1] == self.channels
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            values = x.to(dtype).permute(0, 2, 3, 1)
            values = F.layer_norm(values, (self.channels,), eps=1e-5).permute(0, 3, 1, 2)
            values = self.core(self.down(values))
            values = F.layer_norm(values.permute(0, 2, 3, 1), (self.rank,), eps=1e-5)
            return values.permute(0, 3, 1, 2)

    def forward(self, x):
        return self.up(self.descriptor(x)).to(x.dtype)


class MidstageFocalNet(nn.Module):
    def __init__(self, backbone, arm):
        super().__init__()
        assert arm in ['plain', 'mom_only', *ARMS]
        self.backbone, self.arm = backbone, arm
        self.original_parameter_ids = {id(p) for p in backbone.parameters()}
        if arm.startswith('mom_'):
            backbone.encoder[0][0][0] = C4MomentStem(backbone.encoder[0][0][0], 'moments')
        self.midstage = None
        if arm in ARMS:
            head = backbone.head
            devices = [head.weight.device.index] if head.weight.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                kind = {'mom_focalcontrast': 'contrast', 'focalcontrast_only': 'contrast',
                        'mom_focalraw': 'raw', 'mom_misaligned': 'misaligned',
                        'mom_additive': 'additive'}[arm]
                self.midstage = FocalResidual(kind).to(device=head.weight.device, dtype=head.weight.dtype)

    @property
    def encoder(self): return self.backbone.encoder

    @property
    def head(self): return self.backbone.head

    def forward(self, x):
        for index, layer in enumerate(self.encoder[0]):
            x = layer(x)
            if index == 3 and self.midstage is not None:
                x = x + self.midstage(x)
        for layer in list(self.encoder.children())[1:]:
            x = layer(x)
        return self.head(x)


def counted_flops(model, image, base_counter):
    # The base counter includes all new Conv2d calls; Mom adds its three rotated convs.
    # LN/GELU/centering/gating/elementwise/replicate padding/memory are excluded.
    return moment_flops(model, image, base_counter)
