"""Controlled refinement of the retained C4 moment working structure."""
import torch
from torch import nn
from .c4_moment_stem import C4MomentStem,counted_flops as moment_flops
from .shared_attention import ECA
ARMS=['mean_only','random_moments','moments_eca','aug_moments']

class LibraryNet(nn.Module):
    def __init__(self,backbone,kind):
        super().__init__();assert kind in ['plain','moments','eca_reference',*ARMS]
        self.backbone=backbone;self.kind=kind;self.original_parameter_ids={id(p) for p in backbone.parameters()}
        devices=sorted({p.device.index for p in backbone.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            features=backbone.encoder[0]
            if kind not in ['plain','eca_reference']:
                stem_kind=kind if kind in ['mean_only','random_moments'] else 'moments'
                features[0][0]=C4MomentStem(features[0][0],stem_kind)
            if kind in ['moments_eca','eca_reference']:features[7]=nn.Sequential(features[7],ECA(768,gamma=2,b=1))
    @property
    def encoder(self):return self.backbone.encoder
    @property
    def head(self):return self.backbone.head
    def forward(self,x):
        assert x.ndim==4 and x.shape[1]==3
        return self.backbone(x)

def counted_flops(model,x,base_counter):
    extra=[0];handles=[]
    def hook(module,args,out):extra[0]+=2*out.numel()*(module.in_channels//module.groups)*module.kernel_size[0]
    for module in model.modules():
        if isinstance(module,nn.Conv1d):handles.append(module.register_forward_hook(hook))
    try:base=moment_flops(model,x,base_counter)
    finally:
        for h in handles:h.remove()
    return base+extra[0]
