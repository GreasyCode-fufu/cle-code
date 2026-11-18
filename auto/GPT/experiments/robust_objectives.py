"""Training-only objectives; no dataset relabeling or inference changes.

GCE: Zhang & Sabuncu, NeurIPS 2018.
ELR equations adapted from shengliu66/ELR (MIT, Copyright 2022 Sheng Liu),
blob 752a81cd326ceb930b9e290c4ebd558d8dcb765f. See ELR_LICENSE.txt.
"""
import torch
from torch import nn
import torch.nn.functional as F


class TrainingObjective(nn.Module):
    def __init__(self,kind,num_images=2386,q=.7,beta=.7,strength=3.):
        super().__init__()
        assert kind in {'ce','patient_balance','label_smooth','gce','elr'}
        self.kind=kind
        self.q=q
        self.beta=beta
        self.strength=strength
        if kind=='elr':
            self.register_buffer('target',torch.zeros(num_images,3))
            self.register_buffer('visits',torch.zeros(num_images,dtype=torch.int64))

    def forward(self,logits,labels,indices=None):
        if self.kind=='label_smooth':
            return F.cross_entropy(logits,labels,reduction='none',label_smoothing=.1)
        if self.kind=='gce':
            logp=F.log_softmax(logits.float(),dim=1).gather(1,labels[:,None]).squeeze(1)
            return -torch.expm1(self.q*logp)/self.q
        ce=F.cross_entropy(logits,labels,reduction='none')
        if self.kind!='elr':return ce
        assert indices is not None and indices.unique().numel()==indices.numel()
        p=logits.float().softmax(1).clamp(1e-4,1-1e-4)
        with torch.no_grad():
            normalized=p.detach()/p.detach().sum(1,keepdim=True)
            self.target[indices]=self.beta*self.target[indices]+(1-self.beta)*normalized
            self.visits[indices]+=1
        # The EMA is intentionally not bias-corrected, matching original ELR.
        agreement=(self.target[indices].detach()*p).sum(1)
        return ce+self.strength*torch.log1p(-agreement.clamp(max=1-1e-7))
