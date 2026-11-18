"""Pretraining-preserving first-Swin-merge spatial contrast residual, numerical prototype."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.swin_transformer import PatchMerging,_patch_merging_pad

def merge_descriptor(merged,dim,kind='local_std',norm_eps=1e-5,stat_eps=1e-6):
    assert merged.ndim==4 and merged.shape[-1]==4*dim and kind in ['local_std','mean','global_std']
    dtype=torch.float64 if merged.dtype==torch.float64 else torch.float32
    with torch.autocast(device_type=merged.device.type,enabled=False):
        # Learned LN quadrant-wise affine parameters are deliberately excluded from the statistic.
        u=F.layer_norm(merged.to(dtype),(4*dim,),weight=None,bias=None,eps=norm_eps);cells=u.reshape(*u.shape[:-1],4,dim);mean=cells.mean(-2)
        if kind=='mean':return mean
        std=((cells-mean.unsqueeze(-2)).square().mean(-2)+stat_eps).sqrt()-math.sqrt(stat_eps)
        return std.mean((1,2),keepdim=True).expand_as(std) if kind=='global_std' else std

class MergeContrast(nn.Module):
    def __init__(self,base,kind='local_std'):
        super().__init__();assert type(base) is PatchMerging and isinstance(base.norm,nn.LayerNorm);assert base.norm.normalized_shape==(4*base.dim,) and base.reduction.in_features==4*base.dim and base.reduction.out_features==2*base.dim and base.reduction.bias is None
        self.base=base;self.kind=kind;devices=[base.reduction.weight.device.index] if base.reduction.weight.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            self.adapter=nn.Linear(base.dim,2*base.dim,bias=True,device=base.reduction.weight.device,dtype=base.reduction.weight.dtype);nn.init.zeros_(self.adapter.weight);nn.init.zeros_(self.adapter.bias)
    def forward(self,x):
        merged=_patch_merging_pad(x);original=self.base.reduction(self.base.norm(merged));d=merge_descriptor(merged,self.base.dim,self.kind,self.base.norm.eps)
        return original+self.adapter(d).to(original.dtype)
