"""Small adapters exposing official implementations through CLE-Net's registry API.

The upstream source trees remain unmodified.  Each factory returns one logits tensor
and keeps classifier parameters under the conventional top-level ``head`` name so
the strict ImageNet checkpoint audit in ``train_cle_fair.py`` remains effective.
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent


class _MambaOutPreHead(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.fc1 = original.fc1
        self.act = original.act
        self.norm = original.norm
        self.head_dropout = original.head_dropout

    def forward(self, x):
        return self.head_dropout(self.norm(self.act(self.fc1(x))))


class _CPUBonePreHead(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.in_conv = original.in_conv
        self.global_pool = original.global_pool
        self.flatten = original.flatten
        self.pre_classifier = original.pre_classifier
        self.pool_type = original.pool_type

    def forward(self, x):
        x = self.in_conv(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        if not self.pool_type:
            x = x.permute(0, 2, 3, 1)
        x = self.pre_classifier(x)
        if not self.pool_type:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


def _load_file(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def mambaout_tiny(num_classes=3):
    module = _load_file(
        "cle_external_mambaout",
        ROOT / "MambaOut_CVPR2025" / "models" / "mambaout.py",
    )
    model = module.mambaout_tiny(pretrained=False, num_classes=num_classes)
    original = model.head
    model.pre_head = _MambaOutPreHead(original)
    model.head = original.fc2

    def forward(this, x):
        return this.head(this.pre_head(this.forward_features(x)))

    model.forward = types.MethodType(forward, model)
    return model


def mambavision_t(num_classes=3):
    repo = ROOT / "MambaVision_CVPR2025"
    sys.path.insert(0, str(repo))
    from mambavision.models.mamba_vision import mamba_vision_T
    return mamba_vision_T(pretrained=False, num_classes=num_classes)


def transnext_tiny(num_classes=3):
    repo = ROOT / "TransNeXt_CVPR2024" / "classification"
    sys.path.insert(0, str(repo))
    module = _load_file("cle_external_transnext", repo / "transnext.py")
    return module.transnext_tiny(pretrained=False, num_classes=num_classes)


def cpubone_b2(num_classes=3):
    import timm
    model = timm.create_model(
        "cpubone_b2_bfrobust.r224_in1k",
        pretrained=False,
        num_classes=num_classes,
    )
    original = model.head
    model.pre_head = _CPUBonePreHead(original)
    model.head = original.classifier.linear

    def forward_head(this, x, pre_logits=False):
        x = this.pre_head(x)
        return x if pre_logits else this.head(x)

    model.forward_head = types.MethodType(forward_head, model)
    return model


def thinkingvit_deit(num_classes=3):
    repo = ROOT / "ThinkingViT_CVPR2026"
    sys.path.insert(0, str(repo))
    # ThinkingViT ships a required modified timm.  Force this process to use it;
    # each experiment runs in its own process, so this cannot affect other models.
    for key in [k for k in list(sys.modules) if k == "timm" or k.startswith("timm.")]:
        del sys.modules[key]
    from timm.models import create_model

    model = create_model(
        "thinkingvit",
        pretrained=False,
        num_classes=num_classes,
        thinking_stages=[3, 6],
    )
    original_forward = model.forward

    def entropy(this, logits):
        probs = torch.softmax(logits, dim=1)
        topk = torch.topk(probs, min(10, probs.shape[1]), dim=1)[0]
        return torch.distributions.Categorical(probs=topk).entropy().unsqueeze(1)

    def forward(this, x):
        output = original_forward(x, threshold=0.0)
        if not isinstance(output, tuple):
            return output
        # Training returns (stage0, stage1); evaluation returns (logits, mask).
        return output[-1] if this.training else output[0]

    model.entropy = types.MethodType(entropy, model)
    model.forward = types.MethodType(forward, model)
    return model


REGISTRY = {
    "thinkingvit_deit": ("latest_top_networks.adapters", "thinkingvit_deit"),
    "cpubone_b2": ("latest_top_networks.adapters", "cpubone_b2"),
    "mambavision_t": ("latest_top_networks.adapters", "mambavision_t"),
    "mambaout_tiny": ("latest_top_networks.adapters", "mambaout_tiny"),
    "transnext_tiny": ("latest_top_networks.adapters", "transnext_tiny"),
}
