from pathlib import Path
import ast,os,sys
import torch
from torch.nn import functional as F
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
from qualification_model import QualifiedBackbone
from models.optimize.ac_patient_alignment_window01.library_cle import LibraryNet,counted_flops
from models.optimize.ac_patient_alignment_window01.patient_alignment import contrastive_loss
def main():
 cfg=m.verify();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available();os.nice(10);torch.set_num_threads(2);image=torch.randn(3,3,224,224);initial=None;budgets=[]
 for arm in ['plain','moments',*m.ARMS]:
  m.core.set_seed(45);base=QualifiedBackbone('convnext_tiny',m.Q/'pretrained');state=torch.get_rng_state().clone();original={id(p):p.detach().clone() for p in base.parameters()};model=LibraryNet(base,arm).eval();assert torch.equal(state,torch.get_rng_state()) and all(torch.equal(p,original[id(p)]) for p in model.parameters() if id(p) in original);out=model(image)
  if initial is None:initial=out.detach()
  else:assert torch.equal(out,initial),arm
  assert model.features_for_loss is None
  model.train();out=model(image);loss=F.cross_entropy(out,torch.tensor([0,0,1]));aux=None
  if arm in m.ARMS:
   # Within control gets same-patient positives; cross control gets different patients.
   patients=torch.tensor([0,0,2] if arm=='mom_within' else [0,1,2]);mode='within' if arm=='mom_within' else 'all' if arm=='mom_supcon' else 'cross';aux,_=contrastive_loss(model.projected_features(),torch.tensor([0,0,1]),patients,mode);loss=loss+.05*aux
  loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
  expected=27822435+(0 if arm in ['plain','cross_only'] else 18528)+(106496 if arm in m.ARMS else 0);assert sum(p.numel() for p in model.parameters())==expected
  if arm in m.ARMS:assert float(model.projector[0].weight.grad.norm())>0
  model.eval();budgets.append(dict(arm=arm,parameters=expected,inference_parameters=expected-(106496 if arm in m.ARMS else 0),counted_flops=counted_flops(model,image[:1],m.base_flops),initial_logits_exact=True,RNG_preserved=True,finite_backward=True));del model,base,original
 assert m.sha(m.M/'patient_alignment.py')==m.sha(m.R/'experiments/ac_patient_alignment_prototype_window01/patient_alignment.py')
 # The isolated engine differs only in its training objective callback and logs.
 original=(m.R/'experiments/ac_structure_20261024_window01/run_window01.py').read_text();revised=(S/'objective_engine.py').read_text();a=ast.parse(original);b=ast.parse(revised)
 funcs_a={n.name:ast.dump(n,include_attributes=False) for n in a.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};funcs_b={n.name:ast.dump(n,include_attributes=False) for n in b.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 assert set(funcs_b)-set(funcs_a)=={'compute_training_loss'} and [k for k in funcs_a if funcs_a[k]!=funcs_b[k]]==['worker']
 m.SAVE(S/'module_source_audit.json',dict(passed=True,budgets=budgets,prototype_proof_sha256=m.sha(m.R/'experiments/ac_patient_alignment_prototype_window01/numeric_preflight_cpu.json'),isolated_engine_only_worker_modified=True,network_training=0,CLE_images=0,test_images=0));m.commit('test: verify patient objective public pairing projector gradients CPU loss and isolated engine change scope')
if __name__=='__main__':main()
