"""Bounded cross-scale correlation prototype. No novelty/effectiveness claim."""
import torch
from torch import nn
from torch.nn import functional as F

def bounded_cross_correlation(a,b,eps=1e-4):
    assert a.ndim==b.ndim==3 and a.shape[0]==b.shape[0] and a.shape[-1]==b.shape[-1]
    dtype=torch.float64 if a.dtype==b.dtype==torch.float64 else torch.float32
    with torch.autocast(device_type=a.device.type,enabled=False):
        a=a.to(dtype);b=b.to(dtype)
        a=a-a.mean(-1,keepdim=True);b=b-b.mean(-1,keepdim=True)
        a=a/(a.square().mean(-1,keepdim=True)+eps).sqrt()
        b=b/(b.square().mean(-1,keepdim=True)+eps).sqrt()
        corr=a.bmm(b.transpose(1,2))/a.shape[-1]
        flat=corr.flatten(1)
        # Norm is strictly below1; no division by an arbitrarily small vector norm.
        return flat/(1+flat.square().sum(1,keepdim=True)).sqrt()

class CrossScaleReadout(nn.Module):
    def __init__(self,c3=384,c4=768,rank=16):
        super().__init__();self.rank=rank
        with torch.random.fork_rng(devices=[]):
            self.low=nn.Conv2d(c3,rank,1,bias=False)
            self.high=nn.Conv2d(c4,rank,1,bias=False)
            self.classifier=nn.Linear(rank*rank,3)
            nn.init.zeros_(self.classifier.weight);nn.init.zeros_(self.classifier.bias)
    def forward(self,h3,h4):
        assert h3.shape[-2:]==(h4.shape[-2]*2,h4.shape[-1]*2)
        a=self.low(F.avg_pool2d(h3,2,2)).flatten(2);b=self.high(h4).flatten(2)
        return self.classifier(bounded_cross_correlation(a,b))

class CrossScaleNet(nn.Module):
    def __init__(self,backbone,original_parameter_ids=None):
        super().__init__();self.backbone=backbone
        self.original_parameter_ids=({id(p) for p in backbone.parameters()} if original_parameter_ids is None else original_parameter_ids)
        self.readout=CrossScaleReadout()
    @property
    def encoder(self):return self.backbone.encoder
    @property
    def head(self):return self.backbone.head
    def forward(self,image):
        x=image;h3=None
        for index,stage in enumerate(self.backbone.encoder[0]):
            x=stage(x)
            if index==5:h3=x
        h4=x
        for stage in list(self.backbone.encoder.children())[1:]:x=stage(x)
        return self.backbone.head(x)+self.readout(h3,h4)
