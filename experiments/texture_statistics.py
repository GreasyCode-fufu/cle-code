"""Independent, generic moment descriptors for fixed-feature diagnosis."""
import numpy as np
import torch

def projection():
    q,_=np.linalg.qr(np.random.default_rng(20261024).standard_normal((512,32)))
    return q.astype(np.float32)

def descriptors(feature,project):
    f=feature.float().flatten(2)
    mean=f.mean(-1);center=f-mean.unsqueeze(-1)
    denominator=f.shape[-1]-1
    std=(center.square().sum(-1)/denominator+1e-6).sqrt()
    projected=torch.einsum('cd,bcs->bds',project,center)
    covariance=projected@projected.transpose(1,2)/denominator
    a,b=torch.triu_indices(32,32,device=f.device)
    upper=covariance[:,a,b]
    upper=upper.sign()*(upper.abs()+1e-12).sqrt()
    upper=upper*torch.where(a==b,1.,2.**.5)
    return torch.cat([mean,std,upper],1)

def select_columns(kind):
    return np.arange(512) if kind=='gap' else np.r_[np.arange(512),np.arange(512,1024) if kind=='gap_std' else np.arange(1024,1552)]

def fit_transform(features,indices,kind):
    columns=select_columns(kind);fit=features[np.asarray(indices)][:,columns].astype(np.float64)
    mean=fit.mean(0);scale=np.maximum(fit.std(0),1e-6)
    blocks=[512] if kind=='gap' else [512,len(columns)-512]
    start=0
    for width in blocks:
        scale[start:start+width]*=np.sqrt(width*len(blocks));start+=width
    return (features[:,columns].astype(np.float64)-mean)/scale,columns,mean,scale
