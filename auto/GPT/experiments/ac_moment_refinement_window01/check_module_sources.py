"""CPU check of actual refinement model, initial pairing and preserved data RNG."""
from pathlib import Path
import os,sys
import numpy as np
import torch
from torch.nn import functional as F
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
from qualification_model import QualifiedBackbone,counted_flops as base_flops
from models.optimize.ac_moment_refinement_window01.library_cle import LibraryNet,counted_flops

def main():
    cfg=m.verify();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available();os.nice(10);torch.set_num_threads(2);budgets=[]
    x=torch.randn(1,3,224,224);reference={}
    for kind in ['plain','eca_reference',*m.ARMS]:
        m.core.set_seed(45);base=QualifiedBackbone('convnext_tiny',m.Q/'pretrained');state=torch.get_rng_state().clone();original={id(p):p.detach().clone() for p in base.parameters()};model=LibraryNet(base,kind).eval()
        assert torch.equal(state,torch.get_rng_state()) and all(torch.equal(p,original[id(p)]) for p in model.parameters() if id(p) in original)
        y=model(x)
        if kind in ['plain','eca_reference']:reference[kind]=y.detach()
        else:assert torch.equal(y,reference['eca_reference' if kind=='moments_eca' else 'plain']),kind
        F.cross_entropy(y,torch.tensor([1])).backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        expected=27840968 if kind=='moments_eca' else (27822435 if kind=='plain' else 27822440 if kind=='eca_reference' else 27840963)
        n=sum(p.numel() for p in model.parameters());assert n==expected
        budgets.append(dict(arm=kind,parameters=n,counted_flops=counted_flops(model,x,base_flops),zero_adapter_logits_exact=True,RNG_preserved=True,finite_backward=True));del base,model,original
    # Verify added augmentation leaves original transforms' random sequence unchanged.
    checks=[]
    for arm in m.ARMS:
        for training in [True,False]:
            ids=cfg['indices']['train' if training else 'val'][:8];a=m.q.GuardedData(ids,training,'A');b=m.Data(ids,training,45,arm)
            for pos in range(8):
                torch.manual_seed(900+pos);u,yy,ww,ii=a[pos];st=torch.get_rng_state().clone();torch.manual_seed(900+pos);v,y,w,i=b[pos]
                assert yy==y and ww==w and ii==i and torch.equal(st,torch.get_rng_state())
                assert any(torch.equal(v,torch.rot90(u,k,(-2,-1))) for k in range(4)) if arm=='aug_moments' and training else torch.equal(u,v)
            checks.append(dict(arm=arm,training=training,images=8,RGB_and_RNG_exact=True))
    m.SAVE(S/'module_source_audit.json',dict(passed=True,budgets=budgets,data_checks=checks,network_training=0,test_images=0,scope='CPU computational checks; efficacy unknown until paired retraining'))
    m.commit('test: verify refinement initialization, equal-budget moment controls, finite CPU gradients and augmentation RNG')
if __name__=='__main__':main()
