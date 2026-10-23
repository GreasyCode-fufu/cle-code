"""Round 3 candidates: preserve spatial detail before global aggregation.

Independent PyTorch implementation of shared-weight resolution paths and
SK-style fusion equations. This is not an original SKNet/DRN reproduction.
No third-party library code is imported or copied.
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F


class HighResolutionPath(nn.Module):
    def __init__(self, stage, separate_norm=True, dilation=2):
        super().__init__()
        self.dilation=dilation
        self.separate_norm=separate_norm
        if separate_norm:
            self.norms=nn.ModuleList([nn.ModuleList([copy.deepcopy(b.bn1),copy.deepcopy(b.bn2)]) for b in stage])
            self.projection_norm=copy.deepcopy(stage[0].downsample[1])

    def forward(self, x, stage):
        for i,block in enumerate(stage):
            identity=x
            # Removing the first stride keeps its conv1 rate at 1. Subsequent
            # convolutions compensate the removed stride by dilation 2.
            d1=1 if i==0 else self.dilation
            d2=self.dilation
            n1,n2=self.norms[i] if self.separate_norm else (block.bn1,block.bn2)
            out=F.conv2d(x,block.conv1.weight,stride=1,padding=d1,dilation=d1)
            out=F.relu(n1(out),inplace=False)
            out=F.conv2d(out,block.conv2.weight,stride=1,padding=d2,dilation=d2)
            out=n2(out)
            if block.downsample is not None:
                identity=F.conv2d(x,block.downsample[0].weight,stride=1)
                norm=self.projection_norm if self.separate_norm else block.downsample[1]
                identity=norm(identity)
            x=F.relu(out+identity,inplace=False)
        return x


class ScaleModifier(nn.Module):
    def __init__(self, stage, variant, width=64):
        super().__init__()
        self.variant=variant
        self.enabled=True
        self.capture=False
        self.last_stats={}
        self.last_routing=None
        self.high=HighResolutionPath(stage,separate_norm=variant!='dilated_control',
                                     dilation=1 if variant=='cle_no_dilation' else 2)
        if variant in ('sk_dual','cle_contrast','cle_no_dilation'):
            self.router=nn.Sequential(nn.Conv2d(512,width,1),nn.ReLU(),nn.Conv2d(width,1024,1))
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
        elif variant=='static_dual':
            self.route_logits=nn.Parameter(torch.zeros(1,2,512,1,1))

    def forward(self, x3, stage, low=None):
        high=self.high(x3,stage)
        h=high.float().mean((2,3),keepdim=True)
        if self.variant=='dilated_control':
            self.last_routing=None
            if self.capture:self.last_stats=dict(high_shape=list(high.shape[-2:]))
            return h
        l=low.float().mean((2,3),keepdim=True)
        if self.variant=='fixed_dual':
            route=h.new_full((h.shape[0],2,512,1,1),.5)
        elif self.variant=='static_dual':
            route=self.route_logits.float().softmax(1).expand(h.shape[0],-1,-1,-1,-1)
        else:
            context=h+l
            if self.variant!='sk_dual':
                local_mean=F.avg_pool2d(high.float(),3,stride=1,padding=1,count_include_pad=False)
                contrast=(high.float()-local_mean).abs().mean((2,3),keepdim=True)
                # Relative contrast is bounded; restore descriptor scale without
                # assuming that stronger edges are better tissue evidence.
                relative=(contrast/(h.abs()+1e-4)).clamp(0,2)
                context=context*(1+.5*relative)
            route=self.router(context.to(high.dtype)).float().reshape(-1,2,512,1,1).softmax(1)
        self.last_routing=route.detach()
        fused=route[:,0]*l+route[:,1]*h
        if self.capture:
            self.last_stats=dict(
                high_shape=list(high.shape[-2:]),low_shape=list(low.shape[-2:]),
                high_share=route[:,1].detach().flatten(1).mean(1).cpu().tolist(),
                high_share_channel_sd=route[:,1].detach().flatten(1).std(1).cpu().tolist(),
                descriptor_shift=((fused-l).abs().flatten(1).mean(1)/(l.abs().flatten(1).mean(1)+1e-8)).detach().cpu().tolist(),
            )
        return fused


class ScaleRouteResNet(nn.Module):
    def __init__(self, backbone, variant, width=64):
        super().__init__()
        self.backbone=backbone
        self.modifier=ScaleModifier(backbone.layer4,variant,width)

    def forward(self,x):
        b=self.backbone
        x=b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x=b.layer3(b.layer2(b.layer1(x)))
        if not self.modifier.enabled:
            descriptor=b.avgpool(b.layer4(x))
        else:
            low=None if self.modifier.variant=='dilated_control' else b.layer4(x)
            descriptor=self.modifier(x,b.layer4,low)
        return b.fc(descriptor.flatten(1))
