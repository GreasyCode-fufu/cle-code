"""Unconfirmed local-detail residual, using mature binomial filtering.

Adobe BlurPool 2019 reference, original source, and CC BY-NC-SA 4.0 license
are retained in references. Noncommercial research adaptation, not a full
BlurPool network reproduction or a claim of anti-aliasing/clinical robustness.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .c4_moment_stem import C4MomentStem, counted_flops as moment_flops

ARMS = ['mom_detailcontrast', 'detailcontrast_only', 'mom_detailrawgate',
        'mom_misaligned', 'mom_highpass']


class LocalDetailResidual(nn.Module):
    def __init__(self, kind='contrast', channels=96, spatial=(56, 56)):
        super().__init__()
        assert kind in ['contrast', 'rawgate', 'misaligned', 'highpass']
        self.kind, self.channels = kind, channels
        a = torch.tensor([1., 2., 1.])
        self.register_buffer('kernel', ((a[:, None] * a[None, :])/16)[None, None].repeat(channels, 1, 1, 1))
        if kind == 'misaligned':
            self.register_buffer('permutation', torch.tensor(
                np.random.default_rng(20261025).permutation(spatial[0]*spatial[1]), dtype=torch.long))
        self.up = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def blur(self, x):
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), self.kernel, groups=self.channels)

    def parts(self, x):
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = F.layer_norm(x.to(dtype).permute(0, 2, 3, 1), (self.channels,), eps=1e-5).permute(0, 3, 1, 2)
            detail = z - self.blur(z)
            if self.kind == 'highpass':
                return detail, None
            energy = (self.blur(detail.square()) + 1e-6).sqrt()
            # Every channel has its own spatial gate; this is not a scalar
            # spatial multiplier that the subsequent channel LN would cancel.
            gate = energy/(energy + energy.mean((2, 3), keepdim=True) + 1e-6)
            if self.kind != 'rawgate':
                gate = gate - gate.mean((2, 3), keepdim=True)
            if self.kind == 'misaligned':
                assert x.shape[2]*x.shape[3] == self.permutation.numel()
                gate = gate.flatten(2)[:, :, self.permutation].reshape_as(gate)
            return detail, gate

    def descriptor(self, x):
        assert x.ndim == 4 and x.shape[1] == self.channels
        detail, gate = self.parts(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = detail if gate is None else detail*gate
            return F.layer_norm(value.permute(0, 2, 3, 1), (self.channels,), eps=1e-5).permute(0, 3, 1, 2)

    def forward(self, x):
        return self.up(self.descriptor(x)).to(x.dtype)


class LocalDetailNet(nn.Module):
    def __init__(self, backbone, arm):
        super().__init__()
        assert arm in ['plain', 'mom_only', *ARMS]
        self.backbone, self.arm = backbone, arm
        self.original_parameter_ids = {id(p) for p in backbone.parameters()}
        if arm.startswith('mom_'):
            backbone.encoder[0][0][0] = C4MomentStem(backbone.encoder[0][0][0], 'moments')
        self.detail = None
        if arm in ARMS:
            head = backbone.head
            devices = [head.weight.device.index] if head.weight.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                kind = {'mom_detailcontrast': 'contrast', 'detailcontrast_only': 'contrast',
                        'mom_detailrawgate': 'rawgate', 'mom_misaligned': 'misaligned',
                        'mom_highpass': 'highpass'}[arm]
                self.detail = LocalDetailResidual(kind).to(device=head.weight.device, dtype=head.weight.dtype)

    @property
    def encoder(self): return self.backbone.encoder

    @property
    def head(self): return self.backbone.head

    def forward(self, x):
        for index, layer in enumerate(self.encoder[0]):
            x = layer(x)
            if index == 1 and self.detail is not None:
                x = x + self.detail(x)
        for layer in list(self.encoder.children())[1:]:
            x = layer(x)
        return self.head(x)


def counted_flops(model, image, base_counter):
    # Functional fixed depthwise convolutions bypass the module Conv counter.
    # Count both blur calls (one in highpass), plus Mom's three rotated convs.
    extra = [0]
    handles = []
    def hook(module, args, out):
        extra[0] += (1 if module.kind == 'highpass' else 2)*2*args[0].numel()*9
    for module in model.modules():
        if isinstance(module, LocalDetailResidual):
            handles.append(module.register_forward_hook(hook))
    try:
        base = moment_flops(model, image, base_counter)
    finally:
        for handle in handles: handle.remove()
    return base + extra[0]
