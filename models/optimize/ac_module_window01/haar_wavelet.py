"""Haar/db1-only analytical filter factory; official 2D transforms unchanged."""
import math
import torch
from torch.nn import functional as F
def create_2d_wavelet_filter(wave,in_size,out_size,type=torch.float):
    assert wave in ['db1','haar'], 'Only the preregistered Haar reference is supported'
    lo=torch.tensor([math.sqrt(.5),math.sqrt(.5)],dtype=type)
    hi=torch.tensor([math.sqrt(.5),-math.sqrt(.5)],dtype=type)
    f=torch.stack([lo[None,:]*lo[:,None],lo[None,:]*hi[:,None],hi[None,:]*lo[:,None],hi[None,:]*hi[:,None]])[:,None]
    return f.repeat(in_size,1,1,1),f.repeat(out_size,1,1,1)
def wavelet_2d_transform(x,filters):
    b,c,h,w=x.shape
    return F.conv2d(x,filters,stride=2,groups=c,padding=(filters.shape[2]//2-1,filters.shape[3]//2-1)).reshape(b,c,4,h//2,w//2)
def inverse_2d_wavelet_transform(x,filters):
    b,c,_,h,w=x.shape
    return F.conv_transpose2d(x.reshape(b,c*4,h,w),filters,stride=2,groups=c,padding=(filters.shape[2]//2-1,filters.shape[3]//2-1))
