"""Optical-field support prototypes. No trained CLE result or novelty claim.

The bias-corrected partial-convolution equation is attributed to NVIDIA/Liu
et al. Swin partition/shift semantics follow torchvision (BSD-3-Clause).
The experimental change is bounded residual support adaptation, not a claim
that masked convolution, masked attention, or weighted pooling are new.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def unit_circle(height,width,*,device=None,dtype=torch.float32):
 y=(torch.arange(height,device=device,dtype=dtype)+.5)*2/height-1
 x=(torch.arange(width,device=device,dtype=dtype)+.5)*2/width-1
 return (y[:,None].square()+x[None,:].square()<=1).to(dtype)[None,None]


def resize_support(mask,shape):
 assert mask.ndim==4 and mask.shape[1]==1
 if mask.shape[-2:]!=tuple(shape):mask=F.interpolate(mask.float(),size=shape,mode='area')
 return mask.clamp(0,1)


def partial_conv(base,x,mask,*,cap=None,normalize=True):
 """NVIDIA equation for shared spatial mask; optional experimental ratio cap.

Soft support can arise from area downsampling. It is not an intensity mask.
Count/bias correction stays in FP32 (FP64 for numerical audits), with no
gradient through the supplied support, as in the source method.
"""
 assert isinstance(base,nn.Conv2d) and base.padding_mode=='zeros'
 mask=resize_support(mask,x.shape[-2:]).to(device=x.device,dtype=x.dtype)
 raw=F.conv2d(x*mask,base.weight,base.bias,base.stride,base.padding,base.dilation,base.groups)
 dtype=torch.float64 if x.dtype==torch.float64 else torch.float32
 with torch.no_grad(),torch.autocast(device_type=x.device.type,enabled=False):
  count=F.conv2d(mask.to(dtype),torch.ones((1,1,*base.kernel_size),device=x.device,dtype=dtype),
   stride=base.stride,padding=base.padding,dilation=base.dilation)
  valid=count.clamp(0,1)
  ratio=torch.where(count>0,math.prod(base.kernel_size)/(count+1e-8),torch.zeros_like(count))*valid
  if cap is not None:ratio=ratio.clamp(max=cap)
  if not normalize:ratio=torch.ones_like(count)
 out=raw.to(dtype)
 bias=base.bias.to(dtype)[None,:,None,None] if base.bias is not None else 0.
 out=((out-bias)*ratio+bias)*valid
 return out.to(raw.dtype)


def weighted_pool(x,mask):
 mask=resize_support(mask,x.shape[-2:]).to(device=x.device,dtype=x.dtype)
 dtype=torch.float64 if x.dtype==torch.float64 else torch.float32
 count=mask.to(dtype).sum((-1,-2),keepdim=True)
 numerator=(x.to(dtype)*mask.to(dtype)).sum((-1,-2),keepdim=True)
 return torch.where(count>0,numerator/count.clamp_min(1e-8),torch.zeros_like(numerator)).to(x.dtype)


def _partition(x,wh,ww):
 b,h,w,c=x.shape
 return x.reshape(b,h//wh,wh,w//ww,ww,c).permute(0,1,3,2,4,5).reshape(-1,wh*ww,c)


def window_attention(base,x,mask=None,*,normalize=True):
 """Swin-v1 support-weighted attention using existing QKV/proj parameters.

mask=None reproduces torchvision's ordinary attention including its padded
tokens. An explicit all-one *image* support excludes artificial padding;
this is the no-field padding control, not a claim of exact original output.
All-invalid windows return zero after a finite softmax fallback.
"""
 assert base.attention_dropout==0 and base.dropout==0,'paired RNG requires zero internal attention/projection dropout'
 b,h,w,c=x.shape;wh,ww=base.window_size
 ph=(-h)%wh;pw=(-w)%ww
 padded=F.pad(x,(0,0,0,pw,0,ph));hp,wp=padded.shape[1:3]
 sh,sw=base.shift_size
 if wh>=hp:sh=0
 if ww>=wp:sw=0
 support=None;query=None
 if mask is not None:
  query=resize_support(mask,(h,w)).permute(0,2,3,1).to(x)
  if query.shape[0]==1 and b!=1:query=query.expand(b,-1,-1,-1)
  support=F.pad(query,(0,0,0,pw,0,ph))
 if sh or sw:
  padded=torch.roll(padded,(-sh,-sw),(1,2))
  if support is not None:support=torch.roll(support,(-sh,-sw),(1,2))
 windows=_partition(padded,wh,ww);n=wh*ww;nw=(hp//wh)*(wp//ww)
 qkv=F.linear(windows,base.qkv.weight,base.qkv.bias).reshape(-1,n,3,base.num_heads,c//base.num_heads).permute(2,0,3,1,4)
 q,k,v=qkv.unbind(0)
 logits=(q*(c//base.num_heads)**-.5)@k.transpose(-1,-2)
 logits=logits+base.get_relative_position_bias()
 if sh or sw:
  regions=x.new_zeros((hp,wp));number=0
  for ys in [(0,-wh),(-wh,-sh),(-sh,None)]:
   for xs in [(0,-ww),(-ww,-sw),(-sw,None)]:
    regions[ys[0]:ys[1],xs[0]:xs[1]]=number;number+=1
  region_windows=_partition(regions[None,:,:,None],wh,ww)[...,0]
  different=region_windows[:,:,None]!=region_windows[:,None,:]
  penalty=torch.where(different,x.new_tensor(-100.),x.new_tensor(0.))
  logits=(logits.reshape(b,nw,base.num_heads,n,n)+penalty[None,:,None]).reshape(-1,base.num_heads,n,n)
 if support is not None:
  weights=_partition(support,wh,ww)[...,0]
  if normalize:
   nonempty=weights.sum(-1,keepdim=True)>0
   logw=torch.where(weights>0,weights.clamp_min(torch.finfo(weights.dtype).tiny).log(),torch.full_like(weights,-torch.inf))
   logw=torch.where(nonempty,logw,torch.zeros_like(logw))
   logits=logits+logw[:,None,None,:]
  else:v=v*weights[:,None,:,None]
 attention=logits.softmax(-1)
 output=(attention@v).transpose(1,2).reshape(-1,n,c)
 output=F.linear(output,base.proj.weight,base.proj.bias)
 output=output.reshape(b,hp//wh,wp//ww,wh,ww,c).permute(0,1,3,2,4,5).reshape(b,hp,wp,c)
 if sh or sw:output=torch.roll(output,(sh,sw),(1,2))
 output=output[:,:h,:w].contiguous()
 if query is not None:output=output*(query>0).to(output.dtype)
 return output


class ResidualFieldConv(nn.Module):
 def __init__(self,base,*,cap=4.,normalize=True):
  super().__init__();self.base=base;self.alpha=nn.Parameter(torch.zeros(()));self.cap=cap;self.normalize=normalize
 def forward(self,x,mask):
  original=self.base(x)
  adapted=partial_conv(self.base,x,mask,cap=self.cap,normalize=self.normalize)
  return original+self.alpha.tanh().to(original.dtype)*(adapted-original)


class ResidualFieldAttention(nn.Module):
 def __init__(self,base,*,normalize=True):
  super().__init__();self.base=base;self.alpha=nn.Parameter(torch.zeros(()));self.normalize=normalize
 def forward(self,x,mask):
  original=self.base(x);adapted=window_attention(self.base,x,mask,normalize=self.normalize)
  return original+self.alpha.tanh().to(original.dtype)*(adapted-original)
