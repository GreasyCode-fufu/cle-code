from pathlib import Path
import gc,os,sys
import torch
from torch.nn import functional as F
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
from qualification_model import QualifiedBackbone
from models.optimize.ac_merge_contrast_window01.library_cle import LibraryNet,counted_flops
def main():
 cfg=m.verify();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available();os.nice(10);torch.set_num_threads(2);image=torch.randn(1,3,384,384);initial=None;budgets=[]
 for arm in ['plain',*m.ARMS]:
  m.core.set_seed(45);base=QualifiedBackbone('swin_t',m.Q/'pretrained');rng=torch.get_rng_state().clone();original={id(p):p.detach().clone() for p in base.parameters()};model=LibraryNet(base,arm).eval();assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(p,original[id(p)]) for p in model.parameters() if id(p) in original);out=model(image)
  if initial is None:initial=out.detach()
  else:assert torch.equal(out,initial),arm
  F.cross_entropy(out,torch.tensor([1])).backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
  expected=27521661+(18528 if arm not in ['plain','local_only'] else 0)+(18624 if arm not in ['plain','mom_only'] else 0);count=sum(p.numel() for p in model.parameters());assert count==expected
  if arm not in ['plain','local_only']:assert float(model.encoder[0][0][0].adapter.weight.grad.norm())>0
  if arm not in ['plain','mom_only']:assert float(model.encoder[0][2].adapter.weight.grad.norm())>0
  if arm=='local_only':assert not hasattr(model.encoder[0][0][0],'adapter')
  flops=counted_flops(model,image,m.base_flops);expected_flops=dict(plain=28597871616,mom_only=29192414208,local_only=28682806272,mom_local=29277348864,mom_mean=29277348864,mom_global=29277348864)[arm];assert flops==expected_flops
  budgets.append(dict(arm=arm,parameters=count,counted_flops=flops,initial_logits_exact=True,RNG_preserved=True,original_parameter_ids_preserved=True,finite_backward=True,drop_moment_removes_all18528_parameters=arm=='local_only'));del model,base,original;gc.collect()
 assert m.sha(m.M/'merge_contrast.py')==m.sha(m.R/'experiments/ac_merge_contrast_prototype_window01/merge_contrast.py')
 assert m.sha(m.M/'c4_moment_stem.py')==m.sha(m.R/'models/optimize/ac_module_window01/c4_moment_stem.py')
 m.SAVE(S/'module_source_audit.json',dict(passed=True,budgets=budgets,prototype_numeric_sha256=m.sha(m.R/'experiments/ac_merge_contrast_prototype_window01/numeric_preflight_cpu.json'),five_distinct_controls=True,new_CLE_training=0,CLE_images=0,test_images=0));m.commit('test: verify all5 Swin merge arms public initialization original weights RNG gradients true moment removal and384 budgets')
if __name__=='__main__':main()
