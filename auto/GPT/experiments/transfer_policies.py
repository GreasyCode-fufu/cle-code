"""Training constraints inspired by established transfer-learning controls.

The starting-point anchor follows the L2-SP idea (Li et al., ICML 2018),
added to the unchanged AdamW recipe; this is not an exact paper reproduction.
"""
import torch
from torch import nn

ARMS={'head_only','stage4_only','bn_frozen','starting_anchor'}

def configure(model,kind):
    assert kind in ARMS
    for name,p in model.named_parameters():
        p.requires_grad_(kind not in {'head_only','stage4_only'} or
                         name.startswith('fc.') or (kind=='stage4_only' and name.startswith('layer4.')))

def training_mode(model,kind):
    model.train()
    if kind=='head_only':
        for name,m in model.named_children():
            if name!='fc':m.eval()
    elif kind=='stage4_only':
        for name,m in model.named_children():
            if name not in {'layer4','fc'}:m.eval()
    elif kind=='bn_frozen':
        for m in model.modules():
            if isinstance(m,nn.BatchNorm2d):m.eval()

def reference(model):
    return {k:p.detach().clone() for k,p in model.named_parameters() if not k.startswith('fc.')}

def anchor(model,initial,strength=.01):
    return .5*strength*sum((p.float()-initial[k]).square().sum() for k,p in model.named_parameters() if k in initial)

def drift(model,initial):
    with torch.no_grad():
        square=sum(float((p.float()-initial[k]).square().sum()) for k,p in model.named_parameters() if k in initial)
        count=sum(p.numel() for k,p in model.named_parameters() if k in initial)
        return (square/count)**.5
