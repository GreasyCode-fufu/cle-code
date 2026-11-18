"""Mechanistically distinct faithful module references on frozen ConvNeXt-Tiny.

These module placements are task-specific experiments, not a reproduction of
each paper's full architecture and not established architectural novelty.
"""
import torch
from torch import nn
from .shared_attention import EMA, ECA, SimAM
from .wtconv_reference import WTConv2d
from .c4_moment_stem import C4MomentStem, C4MomentNet

ARMS=['wtconv','simam','eca','ema','c4_moments','c4_aug']

class FaithfulSimAM(nn.Module):
    def __init__(self):
        super().__init__();self.gate=SimAM(e_lambda=1e-4)
    def forward(self,x):
        # Shared PSAA's SimAM returns only weights. Restore official feature output.
        return x*self.gate(x)

class LibraryNet(nn.Module):
    def __init__(self,backbone,kind):
        super().__init__();assert kind in ['plain',*ARMS];self.backbone=backbone;self.kind=kind
        self.original_parameter_ids={id(p) for p in backbone.parameters()}
        devices=sorted({p.device.index for p in backbone.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            features=backbone.encoder[0]
            if kind=='wtconv':
                # Multi-scale hierarchy: last DW7 of stages2/3; copy original Conv
                # objects so pretrained weights and low-rate parameter identity persist.
                for stage in [3,5]:
                    old=features[stage][-1].block[0]
                    assert old.kernel_size==(7,7) and old.groups==old.in_channels
                    new=WTConv2d(old.in_channels,old.out_channels,kernel_size=7,wt_levels=2,wt_type='db1')
                    new.base_conv=old
                    features[stage][-1].block[0]=new
            elif kind=='simam':features[5]=nn.Sequential(features[5],FaithfulSimAM())
            elif kind=='eca':features[7]=nn.Sequential(features[7],ECA(768,gamma=2,b=1))
            elif kind=='ema':features[5]=nn.Sequential(features[5],EMA(384,factor=32))
            elif kind=='c4_moments':features[0][0]=C4MomentStem(features[0][0],'moments')
    @property
    def encoder(self):return self.backbone.encoder
    @property
    def head(self):return self.backbone.head
    def forward(self,x):
        assert x.ndim==4 and x.shape[1]==3
        return self.backbone(x)

def counted_flops(model,image,base_counter):
    extra=[0];handles=[]
    def hook(m,args,out):
        x=args[0]
        if isinstance(m,WTConv2d):
            b,c,h,w=x.shape
            for _ in range(m.wt_levels):
                h+=h%2;w+=w%2
                extra[0]+=16*b*c*h*w  # DWT+IWT, each 4 filters of 2x2, 2FLOP/MAC
                h//=2;w//=2
        elif isinstance(m,nn.Conv1d):extra[0]+=2*out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]
        elif isinstance(m,EMA):extra[0]+=4*x.numel() # two Bg x1xCg @ Bg xCgxHW products
        elif isinstance(m,C4MomentStem):extra[0]+=3*2*out.numel()*m.base.in_channels*16
    for m in model.modules():
        if isinstance(m,(WTConv2d,nn.Conv1d,EMA,C4MomentStem)):handles.append(m.register_forward_hook(hook))
    try:base=base_counter(model,image)
    finally:
        for handle in handles:handle.remove()
    return base+extra[0]
