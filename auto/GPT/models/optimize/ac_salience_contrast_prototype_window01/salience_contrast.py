"""Feature-content attention contrast; mature attention adapted as a zero residual.

Attention equations: Ilse, Tomczak and Welling (ICML2018), official MIT source
retained in references. This does not reproduce their whole MIL architecture.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .c4_moment_stem import C4MomentStem,counted_flops as moment_flops

ARMS=['mom_salience','salience_only','mom_weightedmean','mom_misaligned','mom_ungated']

class SalienceReadout(nn.Module):
    def __init__(self,kind,channels=768,rank=16):
        super().__init__();assert kind in ['contrast','weightedmean','misaligned','ungated']
        self.kind,self.channels,self.rank=kind,channels,rank
        self.v=nn.Linear(channels,rank)
        self.u=nn.Linear(channels,rank) if kind!='ungated' else None
        self.w=nn.Linear(rank,1,bias=False)
        self.classifier=nn.Linear(channels,3)
        nn.init.zeros_(self.classifier.weight);nn.init.zeros_(self.classifier.bias)
        if kind=='misaligned':
            self.register_buffer('permutation',torch.tensor(np.random.default_rng(20261025).permutation(49),dtype=torch.long))
    def weights(self,h):
        z=self.v(h).tanh()
        if self.u is not None:z=z*self.u(h).sigmoid()
        return self.w(z).squeeze(-1).softmax(-1)
    def descriptor(self,x,normalize=True):
        assert x.ndim==4 and x.shape[1]==self.channels
        dtype=torch.float64 if x.dtype==torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type,enabled=False):
            h=x.to(dtype).flatten(2).transpose(1,2)
            score_input=F.layer_norm(h,(self.channels,),eps=1e-5)
            a=self.weights(score_input)
            if self.kind=='misaligned':
                assert h.shape[1]==49
                a=a[:,self.permutation]
            values=h if self.kind=='weightedmean' else h-h.mean(1,keepdim=True)
            d=a[:,None,:].bmm(values).squeeze(1)
            if normalize:d=F.layer_norm(d,(self.channels,),eps=1e-5)
        return d
    def forward(self,x):return self.classifier(self.descriptor(x))

class SalienceNet(nn.Module):
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
                kind={'mom_salience':'contrast','salience_only':'contrast','mom_weightedmean':'weightedmean','mom_misaligned':'misaligned','mom_ungated':'ungated'}[arm]
                self.readout=SalienceReadout(kind).to(device=head.weight.device,dtype=head.weight.dtype)
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
        b,c,h,w=args[0].shape;extra[0]+=2*b*c*h*w
    for m in model.modules():
        if isinstance(m,SalienceReadout):handles.append(m.register_forward_hook(hook))
    try:base=moment_flops(model,image,base_counter)
    finally:
        for h in handles:h.remove()
    return base+extra[0]
