"""Controlled DINOv2 adaptation for CLE; independent implementations.

LoRA follows the published low-rank Q/V update equation. The generic local
adapter is inspired by convolutional bypasses, not a Convpass reproduction.
The context-conditioned local adapter is an experimental design, not a claim
that conditioning, adapters, CLS tokens or FiLM are novel.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ['frozen_head', 'full_finetune', 'lora_qv', 'local_adapter',
            'cle_cls_context', 'cle_mean_context']


class LoRAQKV(nn.Module):
    def __init__(self, base, rank=24):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = 1.0  # alpha=rank
        channels = base.in_features
        assert base.out_features == 3 * channels
        self.aq = nn.Parameter(torch.empty(rank, channels))
        self.bq = nn.Parameter(torch.zeros(channels, rank))
        self.av = nn.Parameter(torch.empty(rank, channels))
        self.bv = nn.Parameter(torch.zeros(channels, rank))
        nn.init.kaiming_uniform_(self.aq, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.av, a=math.sqrt(5))
        self.enabled = True

    def forward(self, x):
        base = self.base(x)
        if not self.enabled:
            return base
        q = F.linear(F.linear(x, self.aq), self.bq) * self.scale
        v = F.linear(F.linear(x, self.av), self.bv) * self.scale
        return base + torch.cat((q, torch.zeros_like(q), v), dim=-1)


class LocalTokenAdapter(nn.Module):
    def __init__(self, channels=384, rank=48, context=None):
        super().__init__()
        assert context in (None, 'cls', 'mean')
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.down = nn.Linear(channels, rank)
        self.spatial = nn.Conv2d(rank, rank, 3, padding=1, groups=rank)
        self.up = nn.Linear(rank, channels)
        self.context_kind = context
        self.context = nn.Linear(channels, 2 * rank) if context else None
        if self.context is not None:
            nn.init.zeros_(self.context.weight)
            nn.init.zeros_(self.context.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.enabled = True

    def forward(self, tokens):
        if not self.enabled:
            return torch.zeros_like(tokens)
        normalized = self.norm(tokens)
        patch = F.gelu(self.down(normalized[:, 1:]))
        batch, count, rank = patch.shape
        side = math.isqrt(count)
        assert side * side == count
        local = self.spatial(patch.transpose(1, 2).reshape(batch, rank, side, side))
        local = local.flatten(2).transpose(1, 2)
        if self.context is not None:
            context = normalized[:, 0] if self.context_kind == 'cls' else normalized[:, 1:].mean(1)
            scale, bias = self.context(context).chunk(2, dim=-1)
            local = local * (1 + .5 * scale.tanh().unsqueeze(1)) + .5 * bias.tanh().unsqueeze(1)
        delta = self.up(F.gelu(local))
        return torch.cat((torch.zeros_like(tokens[:, :1]), delta), dim=1)


def add_local_delta(module, args, output):
    adapter = module.cle_adapter
    return output + adapter(output) if adapter.enabled else output


class CLETokenAdapterNet(nn.Module):
    def __init__(self, root, variant):
        super().__init__()
        from pretrained_representations import dino
        assert variant in VARIANTS
        self.variant = variant
        self.encoder = dino(root)
        self.source_keys = tuple(self.encoder.state_dict())
        # Paired heads must be created before random adapter initializations.
        self.head = nn.Linear(768, 3)
        self.modifiers = []  # modules are registered under encoder.blocks only
        if variant == 'full_finetune':
            self.encoder.requires_grad_(True)
        for block in list(self.encoder.blocks)[8:12]:
            if variant == 'lora_qv':
                block.attn.qkv = LoRAQKV(block.attn.qkv)
                self.modifiers.append(block.attn.qkv)
            elif variant in {'local_adapter', 'cle_cls_context', 'cle_mean_context'}:
                context = {'cle_cls_context': 'cls', 'cle_mean_context': 'mean'}.get(variant)
                block.add_module('cle_adapter', LocalTokenAdapter(rank=24 if context else 48, context=context))
                block.register_forward_hook(add_local_delta)
                self.modifiers.append(block.cle_adapter)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        if self.variant != 'full_finetune':
            self.encoder.eval()
            for modifier in self.modifiers:
                modifier.train(mode)
        return self

    def enable_modifiers(self, enabled):
        for modifier in self.modifiers:
            modifier.enabled = enabled

    def source_state(self):
        current = self.encoder.state_dict()
        return {key: current[key.replace('.attn.qkv.', '.attn.qkv.base.')
                             if self.variant == 'lora_qv' and '.attn.qkv.' in key else key]
                for key in self.source_keys}

    def descriptor(self, image):
        tokens = self.encoder.forward_features(image)
        assert tokens.shape[1:] == (257, 384)
        return torch.cat((tokens[:, 0], tokens[:, 1:].mean(1)), dim=-1)

    def forward(self, image):
        return self.head(self.descriptor(image))


def counted_flops(model, image):
    """Conv/Linear plus attention QK/AV matmuls, 2 FLOPs/MAC, batch counted.

    Excludes normalization, activation, softmax, interpolation and data movement.
    LoRA raw-Parameter matmuls need their own hook, not just Linear hooks.
    """
    from timm.layers.attention import Attention
    total = [0]
    handles = []

    def hook(module, inputs, output):
        if isinstance(module, nn.Linear):
            total[0] += 2 * output.numel() * module.in_features
        elif isinstance(module, nn.Conv2d):
            total[0] += 2 * output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
        elif isinstance(module, LoRAQKV) and module.enabled:
            total[0] += 8 * inputs[0].numel() * module.rank
        elif isinstance(module, Attention):
            batch, count, channels = inputs[0].shape
            total[0] += 4 * batch * count * count * channels

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, LoRAQKV, Attention)):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(image)
    finally:
        for handle in handles:
            handle.remove()
    return total[0]
