"""C4 power-shape complement to retained mean/std; no whole-network equivariance claim."""
import torch
from torch import nn
from torch.nn import functional as F

def power_partition(group,eps=1e-6):
    assert group.ndim==5 and group.shape[1]==4
    dtype=torch.float64 if group.dtype==torch.float64 else torch.float32
    with torch.autocast(device_type=group.device.type,enabled=False):
        r=group.to(dtype);a=(r[:,0]-r[:,2])*.5;b=(r[:,1]-r[:,3])*.5;c=(r[:,0]-r[:,1]+r[:,2]-r[:,3])*.25
        e1=(a.square()+b.square())*.5;e2=c.square();shape=(e2-e1)/(e1+e2+eps)
    return shape,e1,e2

def descriptor(group,kind):
    if kind=='permuted':group=group[:,[0,1,3,2]]
    shape,e1,e2=power_partition(group)
    if kind=='duplicate_std':return ((group-group.mean(1,keepdim=True)).square().mean(1)+1e-6).sqrt()
    if kind=='raw_h2':return e2.clamp_min(1e-12).sqrt()
    assert kind in ['shape','permuted'];return shape

def lifted_responses(base,x,original):
    w=torch.cat([torch.rot90(base.weight,k,(-2,-1)).contiguous() for k in [1,2,3]],0);b=base.bias.repeat(3) if base.bias is not None else None;rest=F.conv2d(x,w,b,stride=4);batch,channels,height,width=original.shape
    group=torch.cat((original[:,None],rest.reshape(batch,3,channels,height,width)),1)
    return group.double() if group.dtype==torch.float64 else group.float()

def make_partition_stem(moment_class):
    class PartitionStem(moment_class):
        def __init__(self,base,kind='shape'):
            assert kind in ['shape','permuted','duplicate_std','raw_h2'];super().__init__(base,'moments');self.partition_kind=kind
            devices=[base.weight.device.index] if base.weight.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                self.partition_adapter=nn.Conv2d(base.out_channels,base.out_channels,1,bias=True,device=base.weight.device,dtype=base.weight.dtype);nn.init.zeros_(self.partition_adapter.weight);nn.init.zeros_(self.partition_adapter.bias)
        def forward(self,x):
            original=self.base(x);group=lifted_responses(self.base,x,original)
            with torch.autocast(device_type=x.device.type,enabled=False):
                mean=group.mean(1);std=((group-mean[:,None]).square().mean(1)+self.eps).sqrt();extra=descriptor(group,self.partition_kind)
            return original+self.adapter(torch.cat((mean,std),1)).to(original.dtype)+self.partition_adapter(extra).to(original.dtype)
    return PartitionStem
