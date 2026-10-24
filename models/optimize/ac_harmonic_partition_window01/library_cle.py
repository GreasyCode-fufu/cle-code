"""Retained moment stem plus controlled C4 power-shape readout."""
import torch
from torch import nn
from .c4_moment_stem import C4MomentStem,counted_flops
from .harmonic_partition import make_partition_stem,lifted_responses,descriptor
ARMS=['mom_shape','shape_only','mom_permuted','mom_duplicate','mom_rawh2']
PartitionStem=make_partition_stem(C4MomentStem)
class ShapeOnlyStem(C4MomentStem):
    def __init__(self,base):
        # group_mean suppresses creation of the18528-param moment adapter; forward is overridden.
        super().__init__(base,'group_mean');devices=[base.weight.device.index] if base.weight.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            self.partition_adapter=nn.Conv2d(base.out_channels,base.out_channels,1,bias=True,device=base.weight.device,dtype=base.weight.dtype);nn.init.zeros_(self.partition_adapter.weight);nn.init.zeros_(self.partition_adapter.bias)
    def forward(self,x):
        original=self.base(x);group=lifted_responses(self.base,x,original);return original+self.partition_adapter(descriptor(group,'shape')).to(original.dtype)
class LibraryNet(nn.Module):
    def __init__(self,backbone,kind):
        super().__init__();assert kind in ['plain','moments',*ARMS];self.backbone=backbone;self.kind=kind;self.original_parameter_ids={id(p) for p in backbone.parameters()};devices=sorted({p.device.index for p in backbone.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            if kind=='moments':backbone.encoder[0][0][0]=C4MomentStem(backbone.encoder[0][0][0],'moments')
            elif kind=='shape_only':backbone.encoder[0][0][0]=ShapeOnlyStem(backbone.encoder[0][0][0])
            elif kind in ARMS:backbone.encoder[0][0][0]=PartitionStem(backbone.encoder[0][0][0],dict(mom_shape='shape',mom_permuted='permuted',mom_duplicate='duplicate_std',mom_rawh2='raw_h2')[kind])
    @property
    def encoder(self):return self.backbone.encoder
    @property
    def head(self):return self.backbone.head
    def forward(self,x):return self.backbone(x)
