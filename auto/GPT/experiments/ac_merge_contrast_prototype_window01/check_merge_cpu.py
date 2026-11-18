from pathlib import Path
import ast,gc,inspect,json,math,os,sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.swin_transformer import PatchMerging,_patch_merging_pad
S=Path(__file__).resolve().parent;R=S.parents[1];Q=R/'experiments/ac_qualification_20261024_window01';sys.path[:0]=[str(S),str(Q),str(R/'experiments'),str(R)]
import run_qualification as q
from qualification_model import QualifiedBackbone,counted_flops as base_flops
from c4_moment_stem import C4MomentStem,counted_flops
from merge_contrast import merge_descriptor,MergeContrast
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available();os.nice(10);torch.set_num_threads(2);q.verify();rng=np.random.default_rng(20261025);a=rng.normal(size=(2,6,8,3));x=torch.tensor(a,dtype=torch.double,requires_grad=True);merged=_patch_merging_pad(x)
 cells=np.stack([a[:,0::2,0::2],a[:,1::2,0::2],a[:,0::2,1::2],a[:,1::2,1::2]],axis=-2);mu=cells.mean((-2,-1),keepdims=True);var=((cells-mu)**2).mean((-2,-1),keepdims=True);u=(cells-mu)/np.sqrt(var+1e-5);means=u.mean(-2);variances=((u-means[...,None,:])**2).mean(-2);std=np.sqrt(variances+1e-6)-.001
 for kind,expected in [('local_std',std),('mean',means),('global_std',np.broadcast_to(std.mean((1,2),keepdims=True),std.shape))]:assert np.max(np.abs(merge_descriptor(merged,3,kind).detach().numpy()-expected))<1e-13
 haar=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],dtype=float)*.5;bands=np.einsum('...qc,kq->...kc',u,haar);assert np.max(np.abs((bands[...,1:,:]**2).sum(-2)/4-variances))<1e-13
 for kind in ['local_std','mean','global_std']:
  rotated=merge_descriptor(_patch_merging_pad(torch.rot90(x,1,(1,2))),3,kind);expected=torch.rot90(merge_descriptor(merged,3,kind),1,(1,2));assert torch.allclose(rotated,expected,atol=1e-12,rtol=1e-12)
 assert not torch.allclose(merge_descriptor(merged,3,'local_std'),merge_descriptor(merged,3,'global_std'))
 example=torch.zeros(1,2,2,2,dtype=torch.double);example[:,0,:,0]=1;example[:,1,:,0]=-1;swapped=example.flip(-1);assert torch.allclose(merge_descriptor(_patch_merging_pad(example),2,'mean'),merge_descriptor(_patch_merging_pad(swapped),2,'mean'));assert not torch.allclose(merge_descriptor(_patch_merging_pad(example),2,'local_std'),merge_descriptor(_patch_merging_pad(swapped),2,'local_std'))
 z=torch.randn(1,2,2,8,dtype=torch.double,requires_grad=True);assert torch.autograd.gradcheck(lambda v:merge_descriptor(v,2,'local_std'),(z,),eps=1e-6,atol=1e-5,rtol=1e-4)
 for scale in [0.,1e-9,1.]:
  flat=torch.full((1,4,4,12),scale,dtype=torch.double,requires_grad=True)
  for kind in ['local_std','mean','global_std']:
   y=merge_descriptor(flat,3,kind);gradient=torch.autograd.grad(y.sum(),flat,retain_graph=True)[0];assert torch.isfinite(y).all() and torch.isfinite(gradient).all()
   if kind!='mean':assert torch.equal(y,torch.zeros_like(y))
 # Original Microsoft layer, restricted to its inspected pure class, versus installed parent.
 tree=ast.parse((S/'upstream/official_microsoft_swin.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PatchMerging');assert all(not isinstance(n,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal)) for n in ast.walk(node));scope={'__name__':'official_reference','nn':nn,'torch':torch,'__builtins__':{'__build_class__':__build_class__,'super':super,'str':str,'AssertionError':AssertionError}};exec(compile(ast.Module(body=[node],type_ignores=[]),'<inspected official PatchMerging>', 'exec'),scope)
 base=PatchMerging(2).double();official=scope['PatchMerging']((4,4),2).double();official.load_state_dict(base.state_dict(),strict=True);v=torch.randn(2,4,4,2,dtype=torch.double,requires_grad=True);w=v.detach().clone().requires_grad_();actual=base(v);reference=official(w.reshape(2,16,2)).reshape_as(actual);assert torch.equal(actual,reference);target=torch.randn_like(actual);g1=torch.autograd.grad((actual-target).square().sum(),[v,*base.parameters()]);g2=torch.autograd.grad((reference-target).square().sum(),[w,*official.parameters()]);assert all(torch.equal(i,j) for i,j in zip(g1,g2))
 try:scope['PatchMerging']((5,7),2)(torch.randn(1,35,2))
 except AssertionError:pass
 else:raise AssertionError('official odd-shape rejection changed')
 for kind in ['local_std','mean','global_std']:
  base=PatchMerging(2).double();original_ids={id(p) for p in base.parameters()};state=torch.get_rng_state().clone();model=MergeContrast(base,kind);assert torch.equal(state,torch.get_rng_state()) and original_ids<={id(p) for p in model.parameters()}
  odd=torch.randn(1,5,7,2,dtype=torch.double);assert torch.equal(model(odd),base(odd)) and torch.isfinite(merge_descriptor(_patch_merging_pad(odd),2,kind)).all()
 model=MergeContrast(PatchMerging(2),'local_std');opt=torch.optim.AdamW(model.parameters(),lr=1e-3);sample=torch.randn(2,4,4,2);target=torch.randn(2,2,2,4)
 for _ in range(2):opt.zero_grad();loss=F.mse_loss(model(sample),target);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
 del model,opt;image=torch.randn(1,3,384,384);initial=None;budgets=[]
 for kind in ['plain','moments','mom_local','local_only','mom_mean','mom_global']:
  q.core.set_seed(45);model=QualifiedBackbone('swin_t',Q/'pretrained').eval();original={id(p):p.detach().clone() for p in model.parameters()};state=torch.get_rng_state().clone()
  if kind not in ['plain','local_only']:model.encoder[0][0][0]=C4MomentStem(model.encoder[0][0][0],'moments')
  if kind not in ['plain','moments']:model.encoder[0][2]=MergeContrast(model.encoder[0][2],{'mom_local':'local_std','local_only':'local_std','mom_mean':'mean','mom_global':'global_std'}[kind])
  assert torch.equal(state,torch.get_rng_state()) and all(torch.equal(p,original[id(p)]) for p in model.parameters() if id(p) in original)
  output=model(image)
  if initial is None:initial=output.detach()
  else:assert torch.equal(output,initial),kind
  expected=27521661+(18528 if kind not in ['plain','local_only'] else 0)+(18624 if kind not in ['plain','moments'] else 0);assert sum(p.numel() for p in model.parameters())==expected;flops=counted_flops(model,image,base_flops);F.cross_entropy(output,torch.tensor([1])).backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());gradient=float(model.encoder[0][2].adapter.weight.grad.norm()) if kind not in ['plain','moments'] else None
  if gradient is not None:assert gradient>0
  budgets.append(dict(kind=kind,parameters=expected,counted_flops=flops,initial_logits_exact=True,original_parameters_and_RNG_preserved=True,new_merge_gradient_norm=gradient));del model,original,output;gc.collect()
 proof=dict(passed=True,NumPy_normalization_mean_variance=True,Haar_power_oracle=True,mean_does_not_determine_contrast=True,local_and_global_distinct=True,even90_descriptor_covariance=True,odd_runtime_pad_preserved_original_rejects_odd=True,double_gradcheck=True,flat_exact_zero_and_gradients_finite=True,original_Microsoft_Torchvision_even_output_and_all_gradients_exact=True,two_synthetic_merge_updates_finite=True,public_Swin384_all6_initial_logits_RNG_parameters_backward=True,budgets=budgets,CLE_images=0,CLE_training=0,new_test_predictions=0,whole_network_equivariance_claim=False,overall_research_complete=False);(S/'numeric_preflight_cpu.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
