"""Round 4: original-pixel global/local views with shared CNN convolutions.

Independent equations, not a reproduction of CSA/TransMIL. Tile labels are
never inferred from the frame label: supervision applies to the fused frame.
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F


class DomainBatchNorm(nn.Module):
    def __init__(self,original):
        super().__init__()
        self.global_bn=original
        self.local_bn=copy.deepcopy(original)
        self.domain=0

    def forward(self,x):
        return self.local_bn(x) if self.domain else self.global_bn(x)


def replace_norms(module):
    for name,child in list(module.named_children()):
        if isinstance(child,nn.BatchNorm2d):setattr(module,name,DomainBatchNorm(child))
        else:replace_norms(child)


def global_view(x):
    return F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)


def local_views(x):
    assert x.shape[-2:]==(448,448)
    return torch.stack([x[...,r:r+224,c:c+224] for r,c in [(0,0),(0,224),(224,0),(224,224)]],1)


class ViewPool(nn.Module):
    def __init__(self,variant,width=64):
        super().__init__()
        self.variant=variant
        self.capture=False
        self.last_stats={}
        self.last_attention=None
        if variant!='tile_fixed':
            self.blend=nn.Sequential(nn.Linear(512,64),nn.ReLU(),nn.Linear(64,512))
            nn.init.zeros_(self.blend[-1].weight);nn.init.zeros_(self.blend[-1].bias)
            if variant=='tile_attention':
                self.a=nn.Linear(512,width,bias=False)
                self.b=nn.Linear(512,width,bias=False)
                self.score=nn.Linear(width,1,bias=False)
            else:
                self.query=nn.Linear(512,width,bias=False)
                self.key=nn.Linear(512,width,bias=False)
                with torch.no_grad():self.key.weight.copy_(self.query.weight)

    def forward(self,global_features,local_features):
        g,l=global_features.float(),local_features.float()
        if self.variant=='tile_fixed':
            attention=l.new_full((len(g),4),.25)
            alpha=g.new_full(g.shape,.25)
        else:
            if self.variant=='tile_attention':
                scores=self.score(torch.tanh(self.a(l))*torch.sigmoid(self.b(l))).float().squeeze(-1)
                logits=4*torch.tanh(scores)
            else:
                source=l.mean(1) if self.variant=='tile_no_context' else g
                q=F.normalize(self.query(source).float(),dim=-1,eps=1e-6)
                k=F.normalize(self.key(l).float(),dim=-1,eps=1e-6)
                logits=4*torch.einsum('bd,bnd->bn',q,k)
            attention=logits.softmax(1)
            alpha=.5*torch.sigmoid(self.blend(g+l.mean(1)).float())
        regional=(attention.unsqueeze(-1)*l).sum(1)
        fused=(1-alpha)*g+alpha*regional
        self.last_attention=attention.detach()
        if self.capture:
            self.last_stats=dict(
                local_share=alpha.detach().mean(1).cpu().tolist(),
                local_attention_entropy=(-(attention*attention.clamp_min(1e-8).log()).sum(1)).detach().cpu().tolist(),
                descriptor_shift=((fused-g).abs().mean(1)/(g.abs().mean(1)+1e-8)).detach().cpu().tolist(),
                local_attention=attention.detach().cpu().tolist(),
            )
        return fused


class MultiViewResNet(nn.Module):
    def __init__(self,backbone,variant,width=64):
        super().__init__()
        self.backbone=backbone
        self.variant=variant
        self.is_tiled=variant.startswith('tile_')
        self.local_enabled=True
        if self.is_tiled:
            replace_norms(self.backbone)
            self.pool=ViewPool(variant,width)

    def canonical_backbone_state(self):
        return {k.replace('.global_bn.','.'):v for k,v in self.backbone.state_dict().items() if '.local_bn.' not in k}

    def set_domain(self,domain):
        for m in self.backbone.modules():
            if isinstance(m,DomainBatchNorm):m.domain=domain

    def encode(self,x,domain=0):
        if self.is_tiled:self.set_domain(domain)
        b=self.backbone
        x=b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x=b.layer4(b.layer3(b.layer2(b.layer1(x))))
        return b.avgpool(x).flatten(1)

    def forward(self,x):
        if self.variant=='single448':return self.backbone.fc(self.encode(x))
        g=self.encode(global_view(x),0)
        if not self.is_tiled or not self.local_enabled:return self.backbone.fc(g)
        tiles=local_views(x)
        l=self.encode(tiles.flatten(0,1),1).reshape(len(x),4,512)
        self.set_domain(0)
        return self.backbone.fc(self.pool(g,l))
