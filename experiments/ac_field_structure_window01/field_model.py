"""Backbone-independent field-support integration prototype; no model selection.

Only the first spatial stem and depthwise spatial mixers are wrapped in CNNs.
Swin-v1 additionally wraps its existing shifted-window attention. The final
pool is the top-level backbone pool, not EfficientNet's internal SE pools.
"""
from pathlib import Path
import math
import torch
from torch import nn
from torchvision.models.swin_transformer import ShiftedWindowAttention
from field_ops_v2 import resize_support,partial_conv,weighted_pool,window_attention

KINDS=['plain','field_pool','reference','fov_residual','no_field','no_local_norm','no_pool']


class SupportContext:
 def __init__(self):self.mask=None;self.cache={}
 def begin(self,mask):self.mask=mask.detach();self.cache={}
 def support(self,size):
  key=tuple(size)
  if key not in self.cache:self.cache[key]=resize_support(self.mask,key)
  return self.cache[key]
 def clear(self):self.mask=None;self.cache={}


class FieldConv(nn.Module):
 def __init__(self,base,context,kind):
  super().__init__();self.base=base;self.context=context;self.kind=kind
  if kind!='reference':self.alpha=nn.Parameter(torch.zeros(()))
 def forward(self,x):
  mask=self.context.support(x.shape[-2:])
  normalized=self.kind!='no_local_norm'
  adapted=partial_conv(self.base,x,mask,cap=None if self.kind=='reference' else 4.,normalize=normalized)
  if self.kind=='reference':return adapted
  original=self.base(x)
  return original+self.alpha.tanh().to(original.dtype)*(adapted-original)


class FieldAttention(nn.Module):
 def __init__(self,base,context,kind):
  super().__init__();self.base=base;self.context=context;self.kind=kind
  if kind!='reference':self.alpha=nn.Parameter(torch.zeros(()))
 def forward(self,x):
  mask=self.context.support(x.shape[1:3])
  adapted=window_attention(self.base,x,mask,normalize=self.kind!='no_local_norm')
  if self.kind=='reference':return adapted
  original=self.base(x)
  return original+self.alpha.tanh().to(original.dtype)*(adapted-original)


class FieldPool(nn.Module):
 def __init__(self,base,context,kind):
  super().__init__();self.base=base;self.context=context;self.kind=kind
  if kind not in ['field_pool','reference']:self.alpha=nn.Parameter(torch.zeros(()))
 def forward(self,x):
  adapted=weighted_pool(x,self.context.support(x.shape[-2:]))
  if self.kind in ['field_pool','reference']:return adapted
  original=self.base(x)
  return original+self.alpha.tanh().to(original.dtype)*(adapted-original)


class FieldModel(nn.Module):
 def __init__(self,backbone,family,kind):
  super().__init__();assert kind in KINDS
  self.backbone=backbone;self.family=family;self.kind=kind;self.context=SupportContext()
  self.operator_paths=[]
  if kind not in ['plain','field_pool']:
   stem=next(m for m in self.backbone.encoder.modules() if isinstance(m,nn.Conv2d) and math.prod(m.kernel_size)>1)
   def install(parent,prefix=''):
    for name,child in list(parent.named_children()):
     path=prefix+'.'+name if prefix else name
     if isinstance(child,nn.Conv2d) and (child is stem or (child.groups==child.in_channels==child.out_channels and math.prod(child.kernel_size)>1)):
      parent.add_module(name,FieldConv(child,self.context,kind));self.operator_paths.append(dict(path=path,type='conv'))
     elif isinstance(child,ShiftedWindowAttention):
      parent.add_module(name,FieldAttention(child,self.context,kind));self.operator_paths.append(dict(path=path,type='attention'))
     else:install(child,path)
   install(self.backbone.encoder)
  if kind not in ['plain','no_pool']:
   pools=[(name,module) for name,module in self.backbone.encoder.named_children() if isinstance(module,nn.AdaptiveAvgPool2d)]
   assert len(pools)==1,'Must replace only the final top-level pool, never internal SE pools'
   name,pool=pools[0];self.backbone.encoder.add_module(name,FieldPool(pool,self.context,kind))

 @property
 def encoder(self):return self.backbone.encoder
 @property
 def head(self):return self.backbone.head

 def forward(self,x):
  assert x.ndim==4 and x.shape[1]==4,'Image plus paired geometric support required; no inferred intensity mask'
  rgb=x[:,:3].contiguous()
  if self.kind=='plain':return self.backbone(rgb)
  support=torch.ones_like(x[:,3:4]) if self.kind=='no_field' else x[:,3:4]
  self.context.begin(support)
  try:return self.backbone(rgb)
  finally:self.context.clear()


def counted_flops(model,image):
 """2 FLOP/MAC Conv/Linear/attention plus extra support-count convolution.

Explicit wrapper hooks count F.conv2d/F.linear calls bypassing child hooks.
Elementwise corrections, interpolation, softmax/norm/pool/activation omitted.
"""
 total=[0];handles=[]
 def attention_cost(base,x):
  b,h,w,c=x.shape;wh,ww=base.window_size;hp=((h+wh-1)//wh)*wh;wp=((w+ww-1)//ww)*ww;n=wh*ww
  return 8*b*hp*wp*c*c+4*b*(hp//wh)*(wp//ww)*n*n*c
 def hook(module,args,out):
  if isinstance(module,nn.Conv2d):total[0]+=2*out.numel()*(module.in_channels//module.groups)*math.prod(module.kernel_size)
  elif isinstance(module,nn.Linear):total[0]+=2*out.numel()*module.in_features
  elif isinstance(module,ShiftedWindowAttention):total[0]+=attention_cost(module,args[0])
  elif isinstance(module,FieldConv):
   base=module.base;total[0]+=2*out.numel()*(base.in_channels//base.groups)*math.prod(base.kernel_size)
   total[0]+=2*out.shape[0]*out.shape[2]*out.shape[3]*math.prod(base.kernel_size)
  elif isinstance(module,FieldAttention):total[0]+=attention_cost(module.base,args[0])
 for m in model.modules():
  if isinstance(m,(nn.Conv2d,nn.Linear,ShiftedWindowAttention,FieldConv,FieldAttention)):handles.append(m.register_forward_hook(hook))
 try:
  with torch.no_grad():model(image)
 finally:
  for h in handles:h.remove()
 return total[0]
