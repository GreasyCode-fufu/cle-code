from pathlib import Path
import gc,os,sys
import torch
from torch.nn import functional as F
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
from qualification_model import QualifiedBackbone
from models.optimize.ac_focal_pairing_dropmom_control_window01.library_cle import LibraryNet,counted_flops

def main():
 cfg=m.verify();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available();torch.set_num_threads(1);image=torch.randn(2,3,224,224);initial=None;budgets=[]
 for arm in ['plain','mom_only','mom_misaligned',*m.ARMS]:
  m.core.set_seed(45);base=QualifiedBackbone('convnext_tiny',m.Q/'pretrained');rng=torch.get_rng_state().clone();original={id(p):p.detach().clone() for p in base.parameters()};model=LibraryNet(base,arm).eval()
  assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(p,original[id(p)]) for p in model.parameters() if id(p) in original)
  out=model(image)
  if initial is None:initial=out.detach()
  else:assert torch.equal(out,initial),arm
  expected=27822435+(18528 if arm.startswith('mom_') else 0)+(11832 if model.midstage is not None else 0);assert sum(p.numel() for p in model.parameters())==expected
  flops=counted_flops(model,image[:1],m.base_flops);assert flops==8909531136+(202309632 if arm.startswith('mom_') else 0)+(18552576 if model.midstage is not None else 0)
  opt=m.optimizer(model);ids=[id(p) for g in opt.param_groups for p in g['params']];assert len(ids)==len(set(ids))==len(list(model.parameters()))
  if model.midstage is not None:assert {id(p) for p in model.midstage.parameters()}<={id(p) for p in opt.param_groups[1]['params']}
  assert opt.param_groups[0]['lr']==1e-5 and opt.param_groups[1]['lr']==1e-4
  records=[]
  for step in range(2):
   opt.zero_grad(set_to_none=True);out=model(image);loss=F.cross_entropy(out,torch.tensor([0,1]));loss.backward();assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
   if model.midstage is not None:
    grads={n:float(p.grad.norm()) for n,p in model.midstage.named_parameters()};assert grads['up.weight']>0
    assert all(v==0. if step==0 else v>0 for n,v in grads.items() if n!='up.weight'),(arm,step,grads)
   else:grads={}
   moment=float(model.encoder[0][0][0].adapter.weight.grad.norm()) if arm.startswith('mom_') else None
   if moment is not None:assert moment>0
   records.append(dict(step=step,CE=float(loss.detach()),midstage_gradients=grads,moment_gradient=moment));norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
  if arm=='misaligned_only':assert not hasattr(model.encoder[0][0][0],'adapter')
  budgets.append(dict(arm=arm,parameters=expected,counted_flops=flops,initial_logits_exact=True,RNG_preserved=True,original_parameter_ids_preserved=True,finite_backward=True,finite_two_updates=True,midstage_in_head_lr_group=model.midstage is not None,gradient_records=records,drop_moment_removes_all18528_parameters=arm=='misaligned_only'));del model,base,original,opt;gc.collect()
 matched=[]
 for seed in m.SEEDS:
  m.core.set_seed(seed);full=LibraryNet(QualifiedBackbone('convnext_tiny',m.Q/'pretrained'),'mom_misaligned');rng=torch.get_rng_state().clone()
  m.core.set_seed(seed);drop=LibraryNet(QualifiedBackbone('convnext_tiny',m.Q/'pretrained'),'misaligned_only')
  assert torch.equal(rng,torch.get_rng_state()) and not hasattr(drop.encoder[0][0][0],'adapter')
  assert full.midstage.core.kind==drop.midstage.core.kind=='misaligned'
  assert all(torch.equal(v,drop.midstage.state_dict()[k]) for k,v in full.midstage.state_dict().items())
  assert torch.equal(full.encoder[0][0][0].base.weight,drop.encoder[0][0][0].weight)
  fullkeys={k:v for k,v in full.state_dict().items() if not k.startswith('backbone.encoder.0.0.0.')}
  assert all(torch.equal(v,drop.state_dict()[k]) for k,v in fullkeys.items())
  assert sum(p.numel() for p in full.parameters())-sum(p.numel() for p in drop.parameters())==18528
  matched.append(dict(seed=seed,identical_all_retained_state=True,identical_fixed784_permutation=True,identical_rng=True,removed_only18528=True));del full,drop;gc.collect()
 m.SAVE(S/'matched_initialization_cpu.json',dict(passed=True,proof=matched,CLE_images=0))
 proto=m.R/'models/optimize/ac_midstage_focal_window01'
 assert all(m.sha(m.M/n)==m.sha(proto/n) for n in ['midstage_focal.py','c4_moment_stem.py'])
 m.SAVE(S/'module_source_audit.json',dict(passed=True,budgets=budgets,original_model_source_sha256=m.sha(proto/'midstage_focal.py'),prototype_operator_byte_exact=True,new_CLE_training=0,CLE_images=0,test_images=0));m.commit('test: verify formal midstage focal CPU two updates exact initializations optimizer groups true Mom removal and budgets')
if __name__=='__main__':main()
