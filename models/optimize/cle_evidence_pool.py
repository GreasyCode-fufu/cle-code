"""Development candidates for context-constrained spatial evidence pooling.

Controls: GeM; gated attention aggregation (Ilse et al., ICML 2018).
The fixed Haar basis is motivated by the frequency separation in library
61/WTConv (Finder et al., ECCV 2024), but this is not a WTConv reproduction.
No disease label or OOF grade is passed to the network at inference.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def haar_bands(x):
    """Orthonormal 2-D Haar; zero-pad odd sizes, keep signed detail bands."""
    if x.shape[-2]%2 or x.shape[-1]%2:
        x=F.pad(x,(0,x.shape[-1]%2,0,x.shape[-2]%2))
    a,b=x[...,0::2,0::2],x[...,0::2,1::2]
    c,d=x[...,1::2,0::2],x[...,1::2,1::2]
    return torch.stack([(a+b+c+d)/2,(-a+b-c+d)/2,(-a-b+c+d)/2,(a-b-c+d)/2],dim=2)


class EvidencePooling(nn.Module):
    def __init__(self,variant,width=64,control_width=64):
        super().__init__()
        self.variant=variant
        self.last_routing=None
        self.last_attention=None
        if variant=='gem_control':
            self.p=nn.Parameter(torch.tensor(3.))
            return
        self.learned_blend=variant!='cle_fixed_blend'
        if self.learned_blend:
            self.blend=nn.Sequential(nn.Conv2d(512,64,1),nn.ReLU(),nn.Conv2d(64,512,1))
            nn.init.zeros_(self.blend[-1].weight)
            nn.init.zeros_(self.blend[-1].bias)
        self.control=variant=='attention_control'
        if self.control:
            self.a=nn.Conv2d(512,control_width,1)
            self.b=nn.Conv2d(512,control_width,1)
            self.score=nn.Conv2d(control_width,1,1,bias=False)
        else:
            self.key4=nn.Conv2d(512,width,1,bias=False)
            self.key3=nn.Conv2d(256,width,1,bias=False)
            self.gate=nn.Conv2d(512,width,1)
            self.score=nn.Conv2d(width,1,1,bias=False)
            self.use_context=variant!='cle_no_context'
            self.use_frequency=variant!='cle_no_frequency'
            if self.use_context:self.query=nn.Conv2d(512,width,1)
            if self.use_frequency:self.frequency_coefficient=nn.Parameter(torch.tensor(.1))

    def forward(self,x4,x3):
        z=x4.float()
        gap=z.mean((2,3),keepdim=True)
        if self.variant=='gem_control':
            p=self.p.clamp(1.,4.)
            pooled=z.clamp_min(1e-6).pow(p).mean((2,3),keepdim=True).pow(1/p)
            self.last_routing=None;self.last_attention=None
            return pooled-gap
        if self.control:
            logits=self.score(torch.tanh(self.a(x4))*torch.sigmoid(self.b(x4))).float()
        else:
            aligned=F.adaptive_avg_pool2d(x3,x4.shape[-2:])
            keys=torch.tanh(self.key4(x4)+self.key3(aligned))
            logits=self.score(keys*torch.sigmoid(self.gate(x4))).float()
            if self.use_context:
                q=torch.tanh(self.query(x4.mean((2,3),keepdim=True)))
                logits=logits+(keys.float()*q.float()).sum(1,keepdim=True)/math.sqrt(keys.shape[1])
            if self.use_frequency:
                bands=haar_bands(x3.float())[:,:,1:]
                energy=(bands.square().mean((1,2),keepdim=False)+1e-8).sqrt().unsqueeze(1)
                energy=F.interpolate(energy,size=x4.shape[-2:],mode='bilinear',align_corners=False)
                standardized=(energy-energy.mean((2,3),keepdim=True))/(energy.std((2,3),keepdim=True,unbiased=False)+1e-4)
                # Sign is learned: high-frequency content is not assumed to be clean.
                logits=logits+self.frequency_coefficient.tanh()*standardized.clamp(-3,3)
        attention=logits.flatten(2).softmax(-1).reshape_as(logits)
        regional=(z*attention).sum((2,3),keepdim=True)
        alpha=.5*torch.sigmoid(self.blend(x4.mean((2,3),keepdim=True))).float() if self.learned_blend else torch.full_like(gap,.25)
        self.last_attention=attention.detach()
        self.last_routing=torch.stack([1-alpha,alpha],dim=1).detach()
        return alpha*(regional-gap)


class PoolModifier(nn.Module):
    def __init__(self,variant,control_width=64):
        super().__init__()
        self.enabled=True
        self.branch=EvidencePooling(variant,control_width=control_width)

    def forward(self,x4,x3):
        # Broadcasting a vector shift keeps the generic branch diagnostic shape;
        # GAP of this tensor is exactly the new pooled descriptor.
        return x4+self.branch(x4,x3) if self.enabled else x4


class EvidenceResNet(nn.Module):
    def __init__(self,backbone,variant,control_width=64):
        super().__init__()
        self.backbone=backbone
        self.modifier=PoolModifier(variant,control_width)

    def forward(self,x):
        b=self.backbone
        x=b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x=b.layer2(b.layer1(x))
        x3=b.layer3(x)
        x4=b.layer4(x3)
        x4=self.modifier(x4,x3)
        return b.fc(b.avgpool(x4).flatten(1))
