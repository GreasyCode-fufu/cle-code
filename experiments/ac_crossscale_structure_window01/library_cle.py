"""Paired low-rank cross-scale statistics; mature bilinear reference retained."""
import torch
from torch import nn
from torch.nn import functional as F
from .c4_moment_stem import C4MomentStem,counted_flops as moment_flops
ARMS=['mom_corr','corr_only','mom_shuffle','mom_bilinear']

def descriptor(a,b,mode='corr',scale=16.,eps=1e-4):
    assert a.ndim==b.ndim==3 and a.shape==b.shape
    dtype=torch.float64 if a.dtype==b.dtype==torch.float64 else torch.float32
    with torch.autocast(device_type=a.device.type,enabled=False):
        a=a.to(dtype);b=b.to(dtype)
        if mode!='bilinear':
            a=a-a.mean(-1,keepdim=True);b=b-b.mean(-1,keepdim=True)
            a=a/(a.square().mean(-1,keepdim=True)+eps).sqrt()
            b=b/(b.square().mean(-1,keepdim=True)+eps).sqrt()
        z=(a.bmm(b.transpose(1,2))/a.shape[-1]).flatten(1)
        if mode=='bilinear':
            # Exact mature BCNNModule signed-root expression, followed by default L2.
            z=(F.relu(z)+1e-5).sqrt()-(F.relu(-z)+1e-5).sqrt()
            return scale*F.normalize(z,dim=1)
        return scale*z/(1+z.square().sum(1,keepdim=True)).sqrt()

class SpatialReadout(nn.Module):
    def __init__(self,mode,rank=16):
        super().__init__();self.mode=mode;self.rank=rank;self.scale=float(rank)
        self.low=nn.Conv2d(384,rank,1,bias=False);self.high=nn.Conv2d(768,rank,1,bias=False)
        self.classifier=nn.Linear(rank*rank,3)
        nn.init.zeros_(self.classifier.weight);nn.init.zeros_(self.classifier.bias)
        self.register_buffer('spatial_permutation',torch.randperm(49,generator=torch.Generator().manual_seed(20261025)))
    def describe(self,h3,h4):
        assert h3.shape[-2:]==(2*h4.shape[-2],2*h4.shape[-1])
        a=self.low(F.avg_pool2d(h3,2,2)).flatten(2);b=self.high(h4).flatten(2)
        if self.mode=='shuffle':
            assert b.shape[-1]==49;b=b.index_select(-1,self.spatial_permutation)
        return descriptor(a,b,self.mode,self.scale)
    def forward(self,h3,h4):return self.classifier(self.describe(h3,h4))

class LibraryNet(nn.Module):
    def __init__(self,backbone,kind):
        super().__init__();assert kind in ['plain','moments',*ARMS]
        self.backbone=backbone;self.kind=kind;self.original_parameter_ids={id(p) for p in backbone.parameters()}
        devices=sorted({p.device.index for p in backbone.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            if kind not in ['plain','corr_only']:
                backbone.encoder[0][0][0]=C4MomentStem(backbone.encoder[0][0][0],'moments')
            self.readout=None if kind in ['plain','moments'] else SpatialReadout('bilinear' if kind=='mom_bilinear' else 'shuffle' if kind=='mom_shuffle' else 'corr').to(next(backbone.parameters()).device)
    @property
    def encoder(self):return self.backbone.encoder
    @property
    def head(self):return self.backbone.head
    def forward(self,image):
        assert image.ndim==4 and image.shape[1]==3
        if self.readout is None:return self.backbone(image)
        x=image;h3=None
        for index,stage in enumerate(self.backbone.encoder[0]):
            x=stage(x)
            if index==5:h3=x
        h4=x
        for stage in list(self.backbone.encoder.children())[1:]:x=stage(x)
        return self.backbone.head(x)+self.readout(h3,h4)

def counted_flops(model,x,base_counter):
    extra=[0];handles=[]
    def hook(module,args,out):extra[0]+=2*args[1].shape[0]*module.rank**2*args[1].shape[-2]*args[1].shape[-1]
    for module in model.modules():
        if isinstance(module,SpatialReadout):handles.append(module.register_forward_hook(hook))
    try:base=moment_flops(model,x,base_counter)
    finally:
        for handle in handles:handle.remove()
    return base+extra[0]
