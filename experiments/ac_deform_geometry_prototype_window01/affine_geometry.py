"""Bounded locally affine sampling adaptation. Novelty/effectiveness unproven."""
import math
import torch
from torch import nn
from torchvision.ops import deform_conv2d

def affine_offsets(raw,kernel_size,permutation=None,return_jacobian=False):
    assert raw.ndim==4 and raw.shape[1]==6 and kernel_size%2==1
    dtype=torch.float64 if raw.dtype==torch.float64 else torch.float32
    with torch.autocast(device_type=raw.device.type,enabled=False):
        p=raw.to(dtype).tanh();dy=.75*p[:,0];dx=.75*p[:,1];theta=(math.pi/8)*p[:,2]
        sy=math.log(1.5)*p[:,3];sx=math.log(1.5)*p[:,4];shear=.25*p[:,5]
        c=theta.cos();s=theta.sin();ey=sy.exp();ex=sx.exp()
        a00=c*ey;a01=c*shear-s*ex;a10=s*ey;a11=s*shear+c*ex
        line=torch.arange(kernel_size,device=raw.device,dtype=dtype)-(kernel_size//2)
        yy,xx=torch.meshgrid(line,line,indexing='ij');yy=yy.flatten()[None,:,None,None];xx=xx.flatten()[None,:,None,None]
        oy=(a00[:,None]-1)*yy+a01[:,None]*xx+dy[:,None]
        ox=a10[:,None]*yy+(a11[:,None]-1)*xx+dx[:,None]
        # Torchvision offset channels are interleaved(y,x) for every kernel point.
        offset=torch.stack((oy,ox),2)
        if permutation is not None:offset=offset.index_select(1,permutation)
        offset=offset.flatten(1,2)
        if return_jacobian:return offset,torch.stack((torch.stack((a00,a01),1),torch.stack((a10,a11),1)),1)
        return offset

def deform_response(x,offset,weight,padding,stride=1,dilation=1):
    dtype=torch.float64 if x.dtype==offset.dtype==weight.dtype==torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type,enabled=False):
        return deform_conv2d(x.to(dtype),offset.to(dtype),weight.to(dtype),bias=None,stride=stride,padding=padding,dilation=dilation,mask=None)

class AffineResidualDepthwise(nn.Module):
    def __init__(self,base,kind='affine'):
        super().__init__();assert kind in ['affine','permuted_affine','free']
        assert base.groups==base.in_channels==base.out_channels and base.kernel_size[0]==base.kernel_size[1] and base.stride==(1,1) and base.dilation==(1,1)
        self.base=base;self.kind=kind;self.kernel_size=base.kernel_size[0];self.offset_channels=2*self.kernel_size**2 if kind=='free' else 6
        devices=sorted({p.device.index for p in base.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            self.predictor=nn.Conv2d(base.in_channels,self.offset_channels,3,padding=1,device=base.weight.device,dtype=base.weight.dtype)
            nn.init.zeros_(self.predictor.weight);nn.init.zeros_(self.predictor.bias)
            self.register_buffer('spatial_permutation',torch.randperm(self.kernel_size**2,generator=torch.Generator().manual_seed(20261025)).to(base.weight.device))
    def offsets(self,raw):
        if self.kind=='free':return raw.double() if raw.dtype==torch.float64 else raw.float()
        return affine_offsets(raw,self.kernel_size,self.spatial_permutation if self.kind=='permuted_affine' else None)
    def forward(self,x):
        original=self.base(x);offset=self.offsets(self.predictor(x))
        # Two identical official responses at zero make initialization bitwise exact,
        # while their encoder/weight gradients cancel at zero. No detached correction.
        delta=deform_response(x,offset,self.base.weight,self.base.padding)-deform_response(x,torch.zeros_like(offset),self.base.weight,self.base.padding)
        return original+delta.to(original.dtype)
