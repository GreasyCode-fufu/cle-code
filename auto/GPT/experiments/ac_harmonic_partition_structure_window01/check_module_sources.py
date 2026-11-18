from pathlib import Path
import gc,os,sys
import torch
from torch.nn import functional as F
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
from qualification_model import QualifiedBackbone
from models.optimize.ac_harmonic_partition_window01.library_cle import LibraryNet,counted_flops,ShapeOnlyStem
def main():
 cfg=m.verify();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available();os.nice(10);torch.set_num_threads(2);image=torch.randn(2,3,224,224);initial=None;budgets=[]
 for arm in ['plain','moments',*m.ARMS]:
  m.core.set_seed(45);base=QualifiedBackbone('convnext_tiny',m.Q/'pretrained');rng=torch.get_rng_state().clone();original={id(p):p.detach().clone() for p in base.parameters()};model=LibraryNet(base,arm).eval();assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(p,original[id(p)]) for p in model.parameters() if id(p) in original);out=model(image)
  if initial is None:initial=out.detach()
  else:assert torch.equal(out,initial),arm
  loss=F.cross_entropy(out,torch.tensor([0,1]));loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
  expected=27822435+(0 if arm in ['plain','shape_only'] else 18528)+(9312 if arm in m.ARMS else 0);count=sum(p.numel() for p in model.parameters());assert count==expected
  if arm in m.ARMS:assert float(model.encoder[0][0][0].partition_adapter.weight.grad.norm())>0
  if arm=='shape_only':assert isinstance(model.encoder[0][0][0],ShapeOnlyStem) and not hasattr(model.encoder[0][0][0],'adapter')
  flops=counted_flops(model,image[:1],m.base_flops);expected_flops=8909531136 if arm=='plain' else 9111840768 if arm=='moments' else 9054038016 if arm=='shape_only' else 9169643520;assert flops==expected_flops
  budgets.append(dict(arm=arm,parameters=count,counted_flops=flops,initial_logits_exact=True,RNG_preserved=True,original_parameter_ids_preserved=True,finite_backward=True,drop_moment_removes_all18528_parameters=arm=='shape_only'));del model,base,original;gc.collect()
 assert m.sha(m.M/'harmonic_partition.py')==m.sha(m.R/'experiments/ac_harmonic_partition_prototype_window01/harmonic_partition.py')
 m.SAVE(S/'module_source_audit.json',dict(passed=True,budgets=budgets,prototype_numeric_sha256=m.sha(m.R/'experiments/ac_harmonic_partition_prototype_window01/numeric_preflight_cpu.json'),five_distinct_controls=True,new_CLE_training=0,CLE_images=0,test_images=0));m.commit('test: verify all5 power-shape controls including true moment removal public pairing gradients and compute budgets')
if __name__=='__main__':main()
