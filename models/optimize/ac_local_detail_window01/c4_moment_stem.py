"""Pretraining-preserving C4 orientation-moment stem prototype.

Rotated filter lifting and group pooling are mature operations. The experiment
is a zero-projection residual moment adapter in a pretrained patch stem, not a
claim that the whole Swin classifier is equivariant or that group pooling is new.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

KINDS=['moments','mean_only','random_moments','group_mean']

def other_weights(weight,kind,permutations):
    if kind=='random_moments':
        flat=weight.flatten(-2)
        return torch.cat([flat[...,permutations[k]].reshape_as(weight) for k in range(3)],0)
    return torch.cat([torch.rot90(weight,k,(-2,-1)).contiguous() for k in [1,2,3]],0)

def group_statistics(x,weight,bias,*,kind='moments',permutations=None,original=None,eps=1e-6):
    assert x.ndim==4 and x.shape[-1]==x.shape[-2] and x.shape[-1]%4==0
    assert weight.shape[-2:]==(4,4)
    if original is None:original=F.conv2d(x,weight,bias,stride=4)
    w=other_weights(weight,kind,permutations);b=bias.repeat(3) if bias is not None else None
    rest=F.conv2d(x,w,b,stride=4);batch,channels,height,width=original.shape
    group=torch.cat((original[:,None],rest.reshape(batch,3,channels,height,width)),1)
    dtype=torch.float64 if group.dtype==torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type,enabled=False):
        group=group.to(dtype);mean=group.mean(1);variance=(group-mean[:,None]).square().mean(1)
        std=(variance+eps).sqrt()
    return mean,std

class C4MomentStem(nn.Module):
    def __init__(self,base,kind='moments',eps=1e-6):
        super().__init__();assert kind in KINDS and isinstance(base,nn.Conv2d)
        assert base.kernel_size==(4,4) and base.stride==(4,4) and base.padding==(0,0) and base.dilation==(1,1) and base.groups==1
        self.base=base;self.kind=kind;self.eps=eps
        rng=np.random.default_rng(20261024)
        self.register_buffer('spatial_permutations',torch.tensor(np.stack([rng.permutation(16) for _ in range(3)]),dtype=torch.long,device=base.weight.device))
        if kind!='group_mean':
            devices=[base.weight.device.index] if base.weight.is_cuda else []
            # Added zero weights must not alter either CPU or CUDA stochastic-depth/data RNG.
            with torch.random.fork_rng(devices=devices):
                self.adapter=nn.Conv2d(2*base.out_channels,base.out_channels,1,bias=True,device=base.weight.device,dtype=base.weight.dtype)
                nn.init.zeros_(self.adapter.weight);nn.init.zeros_(self.adapter.bias)

    def statistics(self,x,original=None):
        return group_statistics(x,self.base.weight,self.base.bias,kind=self.kind,permutations=self.spatial_permutations,original=original,eps=self.eps)

    def forward(self,x):
        original=self.base(x);mean,std=self.statistics(x,original)
        if self.kind=='group_mean':return mean.to(original.dtype)
        descriptor=torch.cat((mean,mean if self.kind=='mean_only' else std),1)
        return original+self.adapter(descriptor).to(original.dtype)

class C4MomentNet(nn.Module):
    def __init__(self,backbone,kind='moments'):
        super().__init__();assert kind in ['plain',*KINDS];self.backbone=backbone;self.kind=kind;self.operator_path=None
        if kind=='plain':return
        stem=next(m for m in backbone.encoder.modules() if isinstance(m,nn.Conv2d))
        def install(parent,prefix=''):
            for name,child in list(parent.named_children()):
                path=prefix+'.'+name if prefix else name
                if child is stem:parent.add_module(name,C4MomentStem(stem,kind));return path
                found=install(child,path)
                if found:return found
            return None
        self.operator_path=install(backbone.encoder);assert self.operator_path is not None
    @property
    def encoder(self):return self.backbone.encoder
    @property
    def head(self):return self.backbone.head
    def forward(self,x):
        assert x.ndim==4 and x.shape[1]==3
        return self.backbone(x)

def counted_flops(model,x,base_counter):
    # base_counter already handles all module Conv/Linear and functional Swin attention.
    # Only the additional3 functional rotated/permuted stem convolutions bypass it.
    extra=[0];handles=[]
    def hook(module,args,out):
        extra[0]+=3*2*out.numel()*module.base.in_channels*math.prod(module.base.kernel_size)
    for module in model.modules():
        if isinstance(module,C4MomentStem):handles.append(module.register_forward_hook(hook))
    try:base=base_counter(model,x)
    finally:
        for handle in handles:handle.remove()
    return base+extra[0]
