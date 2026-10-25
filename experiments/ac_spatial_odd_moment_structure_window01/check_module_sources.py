from pathlib import Path
import gc,os,sys
import torch
from torch.nn import functional as F
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
from qualification_model import QualifiedBackbone
from models.optimize.ac_spatial_odd_moment_window01.library_cle import LibraryNet,counted_flops
def main():
    cfg=m.verify();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available();torch.set_num_threads(1);image=torch.randn(2,3,224,224);initial=None;budgets=[]
    for arm in ['plain','mom_only',*m.ARMS]:
        m.core.set_seed(45);base=QualifiedBackbone('convnext_tiny',m.Q/'pretrained');rng=torch.get_rng_state().clone();original={id(p):p.detach().clone() for p in base.parameters()};model=LibraryNet(base,arm).eval();assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(p,original[id(p)]) for p in model.parameters() if id(p) in original);out=model(image)
        if initial is None:initial=out.detach()
        else:assert torch.equal(out,initial),arm
        F.cross_entropy(out,torch.tensor([0,1])).backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        expected=27822435+(18528 if arm.startswith('mom_') else 0)+(2307 if arm in m.ARMS else 0);assert sum(p.numel() for p in model.parameters())==expected
        if arm in m.ARMS:assert float(model.projection.weight.grad.norm())>0
        if arm=='odd_only':assert not hasattr(model.encoder[0][0][0],'adapter')
        if arm.startswith('mom_'):assert float(model.encoder[0][0][0].adapter.weight.grad.norm())>0
        flops=counted_flops(model,image[:1],m.base_flops);assert flops==8909531136+(202309632 if arm.startswith('mom_') else 0)+(4608 if arm in m.ARMS else 0)
        budgets.append(dict(arm=arm,parameters=expected,counted_flops=flops,initial_logits_exact=True,RNG_preserved=True,original_parameter_ids_preserved=True,finite_backward=True,drop_moment_removes_all18528_parameters=arm=='odd_only'));del model,base,original;gc.collect()
    proto=m.R/'models/optimize/ac_spatial_odd_moment_prototype_window01'
    assert all(m.sha(m.M/n)==m.sha(proto/n) for n in ['spatial_odd_moment.py','c4_moment_stem.py'])
    m.SAVE(S/'module_source_audit.json',dict(passed=True,budgets=budgets,prototype_numeric_sha256=m.sha(m.R/'experiments/ac_spatial_odd_moment_prototype_window01/numeric_preflight_cpu.json'),prototype_operator_byte_exact=True,new_CLE_training=0,CLE_images=0,test_images=0));m.commit('test: verify formal spatial odd controls CPU public pairing true removal gradients budgets and exact prototype copy')
if __name__=='__main__':main()
