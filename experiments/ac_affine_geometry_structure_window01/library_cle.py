"""Retained moment stem plus controlled local sampling geometry."""
import torch
from torch import nn
from .c4_moment_stem import C4MomentStem,counted_flops as moment_flops
from .affine_geometry import AffineResidualDepthwise
ARMS=['mom_affine','affine_only','mom_permuted','mom_dcn']
class LibraryNet(nn.Module):
    def __init__(self,backbone,kind):
        super().__init__();assert kind in ['plain','moments',*ARMS];self.backbone=backbone;self.kind=kind;self.original_parameter_ids={id(p) for p in backbone.parameters()}
        devices=sorted({p.device.index for p in backbone.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            if kind not in ['plain','affine_only']:backbone.encoder[0][0][0]=C4MomentStem(backbone.encoder[0][0][0],'moments')
            if kind in ARMS:
                block=backbone.encoder[0][5][-1].block;block[0]=AffineResidualDepthwise(block[0],'free' if kind=='mom_dcn' else 'permuted_affine' if kind=='mom_permuted' else 'affine')
    @property
    def encoder(self):return self.backbone.encoder
    @property
    def head(self):return self.backbone.head
    def forward(self,image):return self.backbone(image)
def counted_flops(model,x,base_counter):
    extra=[0];handles=[]
    def hook(module,args,out):extra[0]+=4*out.numel()*module.kernel_size**2
    for module in model.modules():
        if isinstance(module,AffineResidualDepthwise):handles.append(module.register_forward_hook(hook))
    try:base=moment_flops(model,x,base_counter)
    finally:
        for h in handles:h.remove()
    return base+extra[0]
