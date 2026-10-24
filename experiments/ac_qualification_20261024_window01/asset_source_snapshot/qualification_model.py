"""Official pretrained mature backbones, identical two-recipe qualification budget."""
from pathlib import Path
import hashlib
import torch
from torch import nn
from torchvision import models

SPECS = {
 'convnext_tiny': ('convnext_tiny', 'https://download.pytorch.org/models/convnext_tiny-983f1562.pth', '983f1562'),
 'efficientnet_v2_s': ('efficientnet_v2_s', 'https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth', 'dd5fe13b'),
 'swin_t': ('swin_t', 'https://download.pytorch.org/models/swin_t-704ceda3.pth', '704ceda3'),
 'convnext_base': ('convnext_base', 'https://download.pytorch.org/models/convnext_base-6075fbad.pth', '6075fbad'),
}


class QualifiedBackbone(nn.Module):
 def __init__(self, family, assets):
  super().__init__();self.family=family
  model=getattr(models,family)(weights=None)
  from torchvision.ops import StochasticDepth
  stochastic=[m for m in model.modules() if isinstance(m,StochasticDepth)]
  maximum=max((m.p for m in stochastic),default=0.)
  if maximum:
   for module in stochastic:module.p*=.1/maximum
  raw=torch.load(Path(assets)/f'{family}.pth',map_location='cpu',weights_only=True)
  model.load_state_dict(raw,strict=True)
  if family.startswith('convnext'):
   self.encoder=nn.Sequential(model.features,model.avgpool,model.classifier[0],nn.Flatten(1))
   dim=model.classifier[-1].in_features
  elif family=='efficientnet_v2_s':
   self.encoder=nn.Sequential(model.features,model.avgpool,nn.Flatten(1),model.classifier[0])
   dim=model.classifier[-1].in_features
  elif family=='swin_t':
   self.encoder=nn.Sequential(model.features,model.norm,model.permute,model.avgpool,model.flatten)
   dim=model.head.in_features
  else:raise ValueError(family)
  self.head=nn.Linear(dim,3)

 def forward(self,x):return self.head(self.encoder(x))


def counted_flops(model,image):
 """Conv/Linear and Swin functional QKV/projection/attention, 2 FLOP/MAC.

 Excludes norm/activation/pooling/bias/softmax/indexing/memory. Swin's forward
 uses F.linear, so its QKV/projection need an explicit attention-module hook.
 """
 from torchvision.models.swin_transformer import ShiftedWindowAttention
 total=[0];handles=[]
 def hook(module,args,output):
  if isinstance(module,nn.Conv2d):
   total[0]+=2*output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
  elif isinstance(module,nn.Linear):total[0]+=2*output.numel()*module.in_features
  elif isinstance(module,ShiftedWindowAttention):
   b,h,w,c=args[0].shape;wh,ww=module.window_size
   ph=((h+wh-1)//wh)*wh;pw=((w+ww-1)//ww)*ww
   n=wh*ww;nw=(ph//wh)*(pw//ww)
   total[0]+=8*b*ph*pw*c*c+4*b*nw*n*n*c
 for module in model.modules():
  if isinstance(module,(nn.Conv2d,nn.Linear,ShiftedWindowAttention)):
   handles.append(module.register_forward_hook(hook))
 try:
  with torch.no_grad():model(image)
 finally:
  for h in handles:h.remove()
 return total[0]
