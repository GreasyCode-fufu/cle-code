"""Retained Mom plus same-layer covariance power readout; mature-statistic study.

The matrix iteration follows the published iSQRT-COV equations. The local
ridge/floor/LN/residual interface is an explicit adaptation, not a full paper
reproduction or a claim of architectural novelty or clinical efficacy.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .c4_moment_stem import C4MomentStem,counted_flops as moment_flops

ARMS=['mom_covsqrt','covsqrt_only','mom_rawcov','mom_diag','mom_channelperm']

def iterative_sqrt(a,steps=5):
    assert a.ndim==3 and a.shape[-1]==a.shape[-2] and steps>=1
    eye=torch.eye(a.shape[-1],dtype=a.dtype,device=a.device)[None]
    trace=a.diagonal(dim1=-2,dim2=-1).sum(-1)[:,None,None]
    y=a/trace;z=eye.expand_as(a)
    for _ in range(steps):
        t=.5*(3*eye-z.bmm(y));y=y.bmm(t);z=t.bmm(z)
    return y*trace.sqrt()

def covariance_descriptor(q,kind='sqrt',normalize=True,permutation=None,floor=None,ridge=1e-4):
    assert q.ndim==4 and kind in ['sqrt','rawcov','diag','channelperm']
    dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
    with torch.autocast(device_type=q.device.type,enabled=False):
        x=q.to(dtype).flatten(2)
        if kind=='channelperm':
            assert permutation is not None and tuple(permutation.shape)==tuple(x.shape[1:])
            x=x.gather(2,permutation[None].expand(x.shape[0],-1,-1))
        c=x-x.mean(-1,keepdim=True);cov=c.bmm(c.transpose(1,2))/c.shape[-1]
        if kind=='diag':cov=torch.diag_embed(cov.diagonal(dim1=-2,dim2=-1))
        if kind!='rawcov':
            eye=torch.eye(c.shape[1],dtype=dtype,device=x.device)[None]
            if floor is None:floor=iterative_sqrt(ridge*eye)
            cov=iterative_sqrt(cov+ridge*eye)-floor.to(dtype)
        index=torch.triu_indices(c.shape[1],c.shape[1],device=x.device)
        d=cov[:,index[0],index[1]]
        scale=torch.ones(index.shape[1],dtype=dtype,device=x.device)
        scale[index[0]!=index[1]]=2.**.5
        d=d*scale
        if normalize:d=F.layer_norm(d,(d.shape[1],),eps=1e-5)
    return d

class CovarianceReadout(nn.Module):
    def __init__(self,kind,rank=16):
        super().__init__();self.kind,self.rank=kind,rank
        self.project=nn.Conv2d(768,rank,1,bias=False)
        self.classifier=nn.Linear(rank*(rank+1)//2,3)
        nn.init.zeros_(self.classifier.weight);nn.init.zeros_(self.classifier.bias)
        self.register_buffer('floor',iterative_sqrt(1e-4*torch.eye(rank)[None]))
        if kind=='channelperm':
            rng=np.random.default_rng(20261025)
            self.register_buffer('permutation',torch.tensor(np.stack([rng.permutation(49) for _ in range(rank)]),dtype=torch.long))
    def forward(self,x):
        return self.classifier(covariance_descriptor(self.project(x),self.kind,permutation=getattr(self,'permutation',None),floor=self.floor))

class CovarianceNet(nn.Module):
    def __init__(self,backbone,arm):
        super().__init__();assert arm in ['plain','mom_only',*ARMS]
        self.backbone,self.arm=backbone,arm
        self.original_parameter_ids={id(p) for p in backbone.parameters()}
        if arm.startswith('mom_'):
            backbone.encoder[0][0][0]=C4MomentStem(backbone.encoder[0][0][0],'moments')
        self.readout=None
        if arm in ARMS:
            head=backbone.head;devices=[head.weight.device.index] if head.weight.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                kind={'mom_covsqrt':'sqrt','covsqrt_only':'sqrt','mom_rawcov':'rawcov','mom_diag':'diag','mom_channelperm':'channelperm'}[arm]
                self.readout=CovarianceReadout(kind).to(device=head.weight.device,dtype=head.weight.dtype)
    @property
    def encoder(self):return self.backbone.encoder
    @property
    def head(self):return self.backbone.head
    def forward(self,x):
        f=self.encoder[0](x);pooled=f
        for layer in list(self.encoder.children())[1:]:pooled=layer(pooled)
        logits=self.head(pooled)
        return logits if self.readout is None else logits+self.readout(f).to(logits.dtype)

def counted_flops(model,image,base_counter):
    extra=[0];handles=[]
    def hook(m,args,out):
        b,_,h,w=args[0].shape;r=m.rank
        extra[0]+=2*b*r*r*h*w
        if m.kind!='rawcov':extra[0]+=5*3*2*b*r**3
    for m in model.modules():
        if isinstance(m,CovarianceReadout):handles.append(m.register_forward_hook(hook))
    try:base=moment_flops(model,image,base_counter)
    finally:
        for handle in handles:handle.remove()
    return base+extra[0]
