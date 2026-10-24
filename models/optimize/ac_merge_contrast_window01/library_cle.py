"""Retained original moment mechanism transferred to Swin B, plus first-merge controls."""
import torch
from torch import nn
from .c4_moment_stem import C4MomentStem,counted_flops
from .merge_contrast import MergeContrast
ARMS=['mom_only','mom_local','local_only','mom_mean','mom_global']
class LibraryNet(nn.Module):
    def __init__(self,backbone,kind):
        super().__init__();assert kind in ['plain',*ARMS];self.backbone=backbone;self.kind=kind;self.original_parameter_ids={id(p) for p in backbone.parameters()};devices=sorted({p.device.index for p in backbone.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            if kind not in ['plain','local_only']:backbone.encoder[0][0][0]=C4MomentStem(backbone.encoder[0][0][0],'moments')
            if kind not in ['plain','mom_only']:backbone.encoder[0][2]=MergeContrast(backbone.encoder[0][2],dict(mom_local='local_std',local_only='local_std',mom_mean='mean',mom_global='global_std')[kind])
    @property
    def encoder(self):return self.backbone.encoder
    @property
    def head(self):return self.backbone.head
    def forward(self,x):return self.backbone(x)
