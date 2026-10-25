"""Bounded odd spatial statistic, with mature moment controls.

Distribution pooling is mature. No claim of novelty, robustness, whole-network
equivariance, lesion localization, or CLE benefit follows from this operator.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .c4_moment_stem import C4MomentStem

ARMS = ['mom_odd', 'odd_only', 'mom_raw3', 'mom_std', 'mom_even']

def spatial_descriptor(x, kind, normalize=True, eps=1e-6):
    assert x.ndim == 4 and kind in ['odd', 'raw3', 'std', 'even']
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        a = x.to(dtype)
        centered = a-a.mean((-2,-1), keepdim=True)
        variance = centered.square().mean((-2,-1), keepdim=True)
        scale = (variance+eps).sqrt()
        z = centered/scale
        if kind == 'std': d = scale.flatten(1)-eps**0.5
        elif kind == 'raw3': d = z.pow(3).mean((-2,-1))
        else: d = torch.tanh(z).pow(3 if kind == 'odd' else 2).mean((-2,-1))
        # Same parameter-free channel normalization for all four descriptors.
        if normalize: d = F.layer_norm(d, (d.shape[1],), eps=1e-5)
    return d

class SpatialOddNet(nn.Module):
    def __init__(self, backbone, arm):
        super().__init__()
        assert arm in ['plain', 'mom_only', *ARMS]
        self.backbone, self.arm = backbone, arm
        self.original_parameter_ids = {id(p) for p in backbone.parameters()}
        if arm.startswith('mom_'):
            stem = backbone.encoder[0][0][0]
            backbone.encoder[0][0][0] = C4MomentStem(stem, 'moments')
        if arm in ARMS:
            self.kind = {'mom_odd':'odd', 'odd_only':'odd', 'mom_raw3':'raw3', 'mom_std':'std', 'mom_even':'even'}[arm]
            head = backbone.head
            devices = [head.weight.device.index] if head.weight.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                self.projection = nn.Linear(head.in_features, head.out_features,
                                            device=head.weight.device, dtype=head.weight.dtype)
                nn.init.zeros_(self.projection.weight)
                nn.init.zeros_(self.projection.bias)
    @property
    def encoder(self): return self.backbone.encoder
    @property
    def head(self): return self.backbone.head
    def forward(self, x):
        features = self.encoder[0](x)
        pooled = features
        for module in list(self.encoder.children())[1:]: pooled = module(pooled)
        logits = self.head(pooled)
        if self.arm in ARMS:
            logits = logits+self.projection(spatial_descriptor(features, self.kind)).to(logits.dtype)
        return logits
