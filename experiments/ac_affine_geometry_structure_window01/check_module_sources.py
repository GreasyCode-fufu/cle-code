from pathlib import Path
import os,sys
import torch
from torch.nn import functional as F
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
from qualification_model import QualifiedBackbone
from models.optimize.ac_affine_geometry_window01.library_cle import LibraryNet,counted_flops
def main():
 cfg=m.verify();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available();os.nice(10);torch.set_num_threads(2);image=torch.randn(1,3,224,224);initial=None;budgets=[]
 for arm in ['plain','moments',*m.ARMS]:
  m.core.set_seed(45);base=QualifiedBackbone('convnext_tiny',m.Q/'pretrained');state=torch.get_rng_state().clone();original={id(p):p.detach().clone() for p in base.parameters()};model=LibraryNet(base,arm).eval();assert torch.equal(state,torch.get_rng_state()) and all(torch.equal(p,original[id(p)]) for p in model.parameters() if id(p) in original);out=model(image)
  if initial is None:initial=out.detach()
  else:assert torch.equal(out,initial),arm
  F.cross_entropy(out,torch.tensor([1])).backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
  expected=27822435+(0 if arm in ['plain','affine_only'] else 18528)+(338786 if arm=='mom_dcn' else 20742 if arm in m.ARMS else 0);assert sum(p.numel() for p in model.parameters())==expected
  if arm in m.ARMS:assert float(model.encoder[0][5][-1].block[0].predictor.weight.grad.norm())>0
  budgets.append(dict(arm=arm,parameters=expected,counted_flops=counted_flops(model,image,m.base_flops),initial_logits_exact=True,RNG_preserved=True,finite_backward=True));del model,base,original
 m.SAVE(S/'module_source_audit.json',dict(passed=True,budgets=budgets,prototype_NumPy_and_gradient_proof_sha256=m.sha(m.R/'experiments/ac_deform_geometry_prototype_window01/numeric_preflight_cpu.json'),network_training=0,CLE_images=0,test_images=0));m.commit('test: verify all affine sampling controls public initialization parameter pairing CPU gradient and compute accounting')
if __name__=='__main__':main()
