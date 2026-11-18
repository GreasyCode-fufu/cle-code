"""Two-lag spatial relations with matched permutation/duplicate/mature controls.

Lag correlation is mature. This is a testable pretrained residual adaptation,
not a claim of a new autocorrelation theorem, clinical structure or efficacy.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .c4_moment_stem import C4MomentStem

ARMS=['mom_lag','lag_only','mom_permuted','mom_duplicate','mom_acf']

def lag_descriptor(x,kind='valid',normalize=True,permutation=None,eps=1e-6):
    assert x.ndim==4 and kind in ['valid','permuted','duplicate','acf']
    dtype=torch.float64 if x.dtype==torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type,enabled=False):
        a=x.to(dtype)
        if kind=='permuted':
            assert permutation is not None and permutation.numel()==a.shape[-2]*a.shape[-1]
            a=a.flatten(-2).index_select(-1,permutation).reshape_as(a)
        centered=a-a.mean((-2,-1),keepdim=True)
        energy=centered.square().sum((-2,-1))
        values=[]
        for lag in ([1,1] if kind=='duplicate' else [1,3]):
            axes=[]
            if a.shape[-1]>lag:axes.append((centered[...,:-lag],centered[...,lag:]))
            if a.shape[-2]>lag:axes.append((centered[...,:-lag,:],centered[...,lag:,:]))
            correlations=[]
            for left,right in axes:
                if kind=='acf':
                    rho=(left*right).sum((-2,-1))/(energy+eps*a.shape[-2]*a.shape[-1])
                else:
                    numerator=(left*right).mean((-2,-1))
                    denominator=(left.square().mean((-2,-1))+eps).sqrt()*(right.square().mean((-2,-1))+eps).sqrt()
                    rho=numerator/denominator
                correlations.append(rho)
            value=torch.stack(correlations).mean(0) if correlations else centered.sum((-2,-1))*0
            values.append(value)
        d=torch.cat(values,1)
        if normalize:d=F.layer_norm(d,(d.shape[1],),eps=1e-5)
    return d

class LagCorrelationNet(nn.Module):
    def __init__(self,backbone,arm):
        super().__init__();assert arm in ['plain','mom_only',*ARMS]
        self.backbone,self.arm=backbone,arm
        self.original_parameter_ids={id(p) for p in backbone.parameters()}
        if arm.startswith('mom_'):
            stem=backbone.encoder[0][0][0]
            backbone.encoder[0][0][0]=C4MomentStem(stem,'moments')
        if arm in ARMS:
            self.kind={'mom_lag':'valid','lag_only':'valid','mom_permuted':'permuted','mom_duplicate':'duplicate','mom_acf':'acf'}[arm]
            head=backbone.head;devices=[head.weight.device.index] if head.weight.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                self.projection=nn.Linear(2*head.in_features,head.out_features,device=head.weight.device,dtype=head.weight.dtype)
                nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)
            if self.kind=='permuted':
                fixed=np.random.default_rng(20261025).permutation(49)
                self.register_buffer('spatial_permutation',torch.tensor(fixed,dtype=torch.long,device=head.weight.device))
    @property
    def encoder(self):return self.backbone.encoder
    @property
    def head(self):return self.backbone.head
    def forward(self,x):
        features=self.encoder[0](x);pooled=features
        for module in list(self.encoder.children())[1:]:pooled=module(pooled)
        logits=self.head(pooled)
        if self.arm in ARMS:
            d=lag_descriptor(features,self.kind,permutation=getattr(self,'spatial_permutation',None))
            logits=logits+self.projection(d).to(logits.dtype)
        return logits
