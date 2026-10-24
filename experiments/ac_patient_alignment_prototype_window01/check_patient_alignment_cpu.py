from pathlib import Path
import importlib.util,json,os,sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
S=Path(__file__).resolve().parent;R=S.parents[1];sys.path[:0]=[str(S),str(R),str(R/'experiments/ac_qualification_20261024_window01')]
from patient_alignment import PatientAlignmentNet,contrastive_loss,pair_masks
def numpy_loss(z,y,g,mode):
    logits=z@z.T/.1;logits-=logits.max(1,keepdims=True);values=[]
    for i in range(len(z)):
        pos=[j for j in range(len(z)) if i!=j and y[i]==y[j] and (mode=='all' or (g[i]!=g[j] if mode=='cross' else g[i]==g[j]))]
        allowed=[j for j in range(len(z)) if i!=j and (y[i]!=y[j] or j in pos)]
        values.append(-float(np.mean([logits[i,j]-np.log(sum(np.exp(logits[i,k]) for k in allowed)) for j in pos])) if pos else 0.)
    return float(np.mean(values))
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available();os.nice(10);torch.set_num_threads(2);torch.manual_seed(45)
    y=torch.tensor([0,0,0,1,1,1,2,2,2]);g=torch.tensor([0,0,1,0,2,2,1,3,3]);z=F.normalize(torch.randn(9,5,dtype=torch.double),dim=1).requires_grad_();proof=[]
    spec=importlib.util.spec_from_file_location('official_supcon',S/'upstream/losses.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    ref=official.SupConLoss(temperature=.1,base_temperature=.1)(z[:,None],y);ours,_=contrastive_loss(z,y,g,'all');torch.testing.assert_close(ref,ours,atol=1e-12,rtol=1e-12);gr=torch.autograd.grad(ref,z,retain_graph=True)[0];go=torch.autograd.grad(ours,z)[0];torch.testing.assert_close(gr,go,atol=1e-12,rtol=1e-12)
    for mode in ['all','cross','within']:
        value,stats=contrastive_loss(z,y,g,mode);assert abs(float(value.detach())-numpy_loss(z.detach().numpy(),y.numpy(),g.numpy(),mode))<1e-12
        assert torch.autograd.gradcheck(lambda a:contrastive_loss(a,y,g,mode)[0],(z,),atol=1e-4,rtol=1e-3)
        order=torch.tensor([4,0,7,1,5,8,2,3,6]);permuted,_=contrastive_loss(z[order],y[order],g[order],mode);renamed,_=contrastive_loss(z,y,23*g+11,mode);assert torch.allclose(value,permuted,atol=1e-12,rtol=1e-12) and torch.equal(value,renamed)
        positive,allowed=pair_masks(y,g,mode);assert not torch.any(positive&(y[:,None]!=y[None,:])) and not torch.any(positive&~allowed) and not allowed.diag().any()
        const=torch.zeros(9,5,dtype=torch.double,requires_grad=True);loss,_=contrastive_loss(const,y,g,mode);loss.backward();assert torch.isfinite(loss) and torch.isfinite(const.grad).all();proof.append(dict(mode=mode,value=float(value.detach()),**stats))
    single=torch.ones(1,5,dtype=torch.double,requires_grad=True);v,_=contrastive_loss(single,torch.tensor([0]),torch.tensor([0]));v.backward();assert v==0 and torch.isfinite(single.grad).all()
    empty,_=contrastive_loss(z,y,torch.zeros_like(g),'cross');assert empty==0
    from qualification_model import QualifiedBackbone
    from models.optimize.ac_module_window01.c4_moment_stem import C4MomentStem
    torch.manual_seed(45);base=QualifiedBackbone('convnext_tiny',R/'experiments/ac_qualification_20261024_window01/pretrained').eval();image=torch.randn(1,3,224,224)
    with torch.no_grad():expected=base(image)
    state=torch.get_rng_state().clone();ids={id(p) for p in base.parameters()};net=PatientAlignmentNet(base,'mom_cross',C4MomentStem).eval();assert torch.equal(state,torch.get_rng_state()) and ids<={id(p) for p in net.parameters()}
    with torch.no_grad():assert torch.equal(net(image),expected)
    assert sum(p.numel() for p in net.parameters())==27947459
    net.train();logits=net(torch.randn(3,3,224,224));real_aux,_=contrastive_loss(net.projected_features(),torch.tensor([0,0,1]),torch.tensor([0,1,2]),'cross');objective=F.cross_entropy(logits,torch.tensor([0,0,1]))+.05*real_aux;objective.backward()
    assert torch.isfinite(objective) and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()) and float(net.projector[0].weight.grad.norm())>0
    # Small synthetic feature classification objective; no CLE image or training.
    features=torch.randn(9,768);projector=net.projector;optimizer=torch.optim.AdamW(projector.parameters(),lr=1e-4)
    for _ in range(2):optimizer.zero_grad();loss,_=contrastive_loss(F.normalize(projector(features),dim=1),y,g,'cross');loss.backward();assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in projector.parameters());optimizer.step()
    result=dict(passed=True,official_singleview_SupCon_loss_and_gradient=True,NumPy_three_masks_and_losses=True,double_gradcheck=True,mixed_label_patient_mask=True,renaming_and_permutation_invariance=True,empty_and_singleton_finite=True,constant_features_finite=True,public_Tiny_Mom_initial_logits_and_RNG_exact=True,public_finite_backward_and_projection_gradient=True,original_parameter_ids_preserved=True,projector_parameters=106496,full_training_parameters=27947459,synthetic_two_updates_finite=True,loss_modes=proof,CLE_images=0,CLE_neural_training=0,test_predictions=0,overall_research_complete=False);(S/'numeric_preflight_cpu.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
