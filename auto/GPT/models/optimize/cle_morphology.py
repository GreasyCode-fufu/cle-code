"""CLE morphology candidates, independently implemented for controlled research.

The reference follows LSNet (Wang et al., CVPR 2025) LKP/SKA equations:
https://github.com/THU-MIG/lsnet/blob/master/model/lsnet.py
https://github.com/THU-MIG/lsnet/blob/master/model/ska.py
Reference blobs: 1c46507d7b8a84d7fc2631840a2ad421a2663799,
ee0ef7ac261ec8821827f781c712431a68ae7b46.

The shared library's 122 LSConv is an adapted per-channel implementation;
official SKA shares kernel weights with channel index modulo C/8 instead.
No upstream Triton source is vendored here. This implementation uses unfold.
The CLE branch/fusion below is an experimental hypothesis, not verified novelty.
"""
import torch
from torch import nn
import torch.nn.functional as F


def cbr(a, b, kernel=1, groups=1):
    return nn.Sequential(nn.Conv2d(a, b, kernel, padding=kernel//2, groups=groups, bias=False),
                         nn.BatchNorm2d(b), nn.ReLU(inplace=False))


def shared_aggregate(x, weights, kernel=3):
    """SKA's exact ci % weight_channels mapping, not contiguous groups."""
    b, c, h, w = x.shape
    wc = weights.shape[1]
    assert c % wc == 0 and weights.shape == (b, wc, kernel*kernel, h, w)
    patches = F.unfold(x, kernel, padding=kernel//2).reshape(b, c, kernel*kernel, h, w)
    expanded = weights.repeat(1, c//wc, 1, 1, 1)
    return (patches * expanded).sum(2)


class KernelPredictor(nn.Module):
    def __init__(self, channels, large_kernel=7, group_width=8):
        super().__init__()
        assert channels % group_width == 0
        hidden = channels//2
        self.features = nn.Sequential(cbr(channels, hidden),
                                      nn.Conv2d(hidden, hidden, large_kernel,
                                                padding=large_kernel//2, groups=hidden, bias=False),
                                      nn.BatchNorm2d(hidden), cbr(hidden, hidden))
        self.weight_channels = channels//group_width
        self.output = nn.Conv2d(hidden, self.weight_channels*9, 1)
        self.norm = nn.GroupNorm(self.weight_channels, self.weight_channels*9)

    def forward(self, x, positive=False):
        b, _, h, w = x.shape
        z = self.norm(self.output(self.features(x))).reshape(b, self.weight_channels, 9, h, w)
        return z.softmax(2) if positive else z


class LSReference(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.predictor = KernelPredictor(channels)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.norm(shared_aggregate(x, self.predictor(x)))


class PlainControl(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.reduce = cbr(channels, width)
        self.local = cbr(width, width, 3, groups=width)
        self.context = cbr(width, width, 7, groups=width)
        self.project = nn.Sequential(nn.Conv2d(width, channels, 1, bias=False),nn.BatchNorm2d(channels))

    def forward(self, x):
        z = self.reduce(x)
        return self.project(.5*self.local(z) + .5*self.context(z))


class CLEContext(nn.Module):
    """Separate local contrast from context, with bounded dynamic aggregation.

    The weights are conditioned only on current features, never on true labels
    or OOF grades. They are not asserted to be clinical image-quality scores.
    """
    def __init__(self, channels, width=64, local=True, routing=True, contrast=True):
        super().__init__()
        self.reduce = cbr(channels, width)
        self.local_on = local
        self.routing = routing
        self.contrast = contrast
        self.local = cbr(width, width, 3, groups=width)
        self.predictor = KernelPredictor(width)
        self.context_norm = nn.Sequential(nn.BatchNorm2d(width),nn.ReLU(inplace=False))
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(width, max(width//4,8),1),
                                    nn.ReLU(inplace=False),nn.Conv2d(max(width//4,8),2*width,1))
        # Equal mixtures at initialization. Branch paths themselves are nonzero.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.project = nn.Sequential(nn.Conv2d(width,channels,1,bias=False),nn.BatchNorm2d(channels))
        self.last_routing = None

    def forward(self,x):
        z=self.reduce(x)
        local_input=z-F.avg_pool2d(z,3,stride=1,padding=1,count_include_pad=False) if self.contrast else z
        local=self.local(local_input)
        context=self.context_norm(shared_aggregate(z,self.predictor(z,positive=True)))
        if not self.local_on:
            mixed=context
            self.last_routing=None
        elif self.routing:
            route=self.router(z).reshape(z.shape[0],2,z.shape[1],1,1).softmax(1)
            self.last_routing=route.detach()
            mixed=route[:,0]*local+route[:,1]*context
        else:
            self.last_routing=None
            mixed=.5*local+.5*context
        return self.project(mixed)


class MorphologyBlock(nn.Module):
    def __init__(self,channels,variant,control_width=64):
        super().__init__()
        self.enabled=True
        self.scale=.1  # common fixed residual scale for every compared branch
        if variant=='conv_control':self.branch=PlainControl(channels,control_width)
        elif variant=='ls_reference':self.branch=LSReference(channels)
        elif variant=='cle_context':self.branch=CLEContext(channels)
        elif variant=='cle_no_local':self.branch=CLEContext(channels,local=False)
        elif variant=='cle_fixed_mix':self.branch=CLEContext(channels,routing=False)
        elif variant=='cle_no_contrast':self.branch=CLEContext(channels,contrast=False)
        else:raise ValueError(variant)

    def forward(self,x):
        return x+self.scale*self.branch(x) if self.enabled else x


class MorphologyResNet(nn.Module):
    def __init__(self,backbone,variant='cle_context',position=3,control_width=64):
        super().__init__()
        self.backbone=backbone
        self.position=position
        self.modifier=MorphologyBlock({2:128,3:256,4:512}[position],variant,control_width)

    def forward(self,x):
        b=self.backbone
        x=b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x=b.layer1(x)
        x=b.layer2(x)
        if self.position==2:x=self.modifier(x)
        x=b.layer3(x)
        if self.position==3:x=self.modifier(x)
        x=b.layer4(x)
        if self.position==4:x=self.modifier(x)
        return b.fc(b.avgpool(x).flatten(1))
