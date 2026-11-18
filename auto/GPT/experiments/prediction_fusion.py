"""Inner-only bounded temperature scaling and fixed probability mixing."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp,softmax

SCHEMES={
    'resnet_bn_train':['resnet_bn_train'],
    'resnet_bn_frozen':['resnet_bn_frozen'],
    'dino_full':['dino_full'],
    'mix_train_dino':['resnet_bn_train','dino_full'],
    'mix_frozen_dino':['resnet_bn_frozen','dino_full'],
    'mix_resnets':['resnet_bn_train','resnet_bn_frozen'],
}
COMPARISONS=[('mix_frozen_dino','dino_full'),('mix_frozen_dino','resnet_bn_frozen'),
             ('mix_frozen_dino','mix_resnets'),('mix_train_dino','dino_full'),
             ('mix_train_dino','resnet_bn_train')]


def nll(beta,logits,labels):
    logits=np.asarray(logits,dtype=np.float64);labels=np.asarray(labels,dtype=np.int64)
    scaled=beta*logits
    return float(np.mean(logsumexp(scaled,axis=1)-scaled[np.arange(len(labels)),labels]))


def beta_gradient(beta,logits,labels):
    logits=np.asarray(logits,dtype=np.float64);labels=np.asarray(labels,dtype=np.int64)
    return float(np.mean((softmax(beta*logits,axis=1)*logits).sum(1)-logits[np.arange(len(labels)),labels]))


def fit_temperature(inner_logits,inner_labels):
    assert np.isfinite(inner_logits).all() and len(inner_logits)==len(inner_labels)>0
    result=minimize_scalar(nll,args=(inner_logits,inner_labels),bounds=(.05,20),method='bounded',
                           options=dict(xatol=1e-10,maxiter=500))
    assert result.success and .05<=result.x<=20
    beta=min([float(result.x),.05,1.,20.],key=lambda b:nll(b,inner_logits,inner_labels))
    before=nll(1.,inner_logits,inner_labels);after=nll(beta,inner_logits,inner_labels)
    assert after<=min(before,nll(.05,inner_logits,inner_labels),nll(20.,inner_logits,inner_labels))+1e-8
    return dict(beta=beta,temperature=float(1/beta),inner_nll_before=before,inner_nll_after=after,
                beta_gradient=beta_gradient(beta,inner_logits,inner_labels),evaluations=int(result.nfev)+3)


def probabilities(logits,beta):
    assert beta>0
    return softmax(np.asarray(logits,dtype=np.float64)*beta,axis=1)


def mixture(parts):
    assert len(parts) in [1,2]
    return np.mean(np.stack(parts),axis=0)


def probability_scores(prob,labels):
    labels=np.asarray(labels,dtype=np.int64)
    return dict(nll=float(-np.log(np.maximum(prob[np.arange(len(labels)),labels],1e-300)).mean()),
                brier=float(np.square(prob-np.eye(3)[labels]).sum(1).mean()))
