"""CPU mathematical/pretrained integration checks, no CLE model training."""
from pathlib import Path
import copy,hashlib,json,sys
import numpy as np
import torch
from torch import nn
S=Path(__file__).resolve().parent;R=S.parents[1];Q=R/'experiments/ac_qualification_20261024_window01'
sys.path[:0]=[str(S),str(Q)]
from c4_moment_stem import C4MomentStem,C4MomentNet,group_statistics,counted_flops
from qualification_model import QualifiedBackbone,counted_flops as original_counter
torch.set_num_threads(1);torch.manual_seed(20261024)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def oracle(x,w,b):
    outputs=[]
    for k in range(4):
        wk=np.rot90(w,k,axes=(-2,-1));out=np.zeros((len(x),len(w),x.shape[-2]//4,x.shape[-1]//4))
        for batch in range(len(x)):
            for y in range(out.shape[-2]):
                for z in range(out.shape[-1]):out[batch,:,y,z]=(wk*x[batch,:,4*y:4*y+4,4*z:4*z+4]).sum((1,2,3))+(b if b is not None else 0)
        outputs.append(out)
    values=np.stack(outputs,1);return values.mean(1),np.sqrt(((values-values.mean(1,keepdims=True))**2).mean(1)+1e-6)

checks=[]
for bias in [False,True]:
    base=nn.Conv2d(3,5,4,4,bias=bias).double();x=torch.randn(2,3,16,16,dtype=torch.float64)
    stem=C4MomentStem(base).double();mu,std=stem.statistics(x)
    a,b=oracle(x.numpy(),base.weight.detach().numpy(),base.bias.detach().numpy() if base.bias is not None else None)
    assert np.max(np.abs(mu.detach().numpy()-a))<1e-12 and np.max(np.abs(std.detach().numpy()-b))<1e-12
    for k in [1,2,3]:
        ma,sa=stem.statistics(torch.rot90(x,k,(-2,-1)).contiguous())
        assert torch.allclose(ma,torch.rot90(mu,k,(-2,-1)),atol=1e-12,rtol=0)
        assert torch.allclose(sa,torch.rot90(std,k,(-2,-1)),atol=1e-12,rtol=0)
    assert torch.equal(stem(x),base(x))
    random=C4MomentStem(copy.deepcopy(base),'random_moments').double();rm,rs=random.statistics(x);tm,ts=random.statistics(torch.rot90(x,1,(-2,-1)).contiguous())
    assert float((tm-torch.rot90(rm,1,(-2,-1))).abs().max())>1e-5
    flat=torch.ones_like(x);fm,fs=stem.statistics(flat);loss=fs.sum();loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in stem.parameters())
    checks.append(dict(bias=bias,numpy_oracle=True,C4_mean_std_spatial_equivariance=True,zero_projection_exact=True,random_control_breaks_geometric_equivariance=True,flat_patch_std_gradient_finite=True))

x=torch.randn(1,2,8,8,dtype=torch.float64,requires_grad=True);w=torch.randn(2,2,4,4,dtype=torch.float64,requires_grad=True);b=torch.randn(2,dtype=torch.float64,requires_grad=True)
assert torch.autograd.gradcheck(lambda a,c,d:torch.cat(group_statistics(a,c,d),1),(x,w,b),eps=1e-6,atol=1e-5,rtol=1e-4,fast_mode=True)
try:group_statistics(torch.zeros(1,2,10,10),w,b)
except AssertionError:pass
else:raise AssertionError('misaligned stride grid must not claim exact rotation property')
base=nn.Conv2d(3,6,4,4).double();state=torch.get_rng_state();stem=C4MomentStem(base).double();assert torch.equal(state,torch.get_rng_state())
with torch.no_grad():stem.adapter.weight.fill_(.01)
x=torch.randn(1,3,16,16,dtype=torch.float64);mu,std=stem.statistics(x);stem.adapter(torch.cat((mu,std),1)).square().mean().backward()
assert torch.isfinite(base.weight.grad).all() and float(base.weight.grad.abs().sum())>0

torch.manual_seed(45);baseline=QualifiedBackbone('swin_t',Q/'pretrained').eval();x=torch.randn(1,3,224,224)
with torch.no_grad():expected=baseline(x)
original=sum(p.numel() for p in baseline.parameters());budgets=[]
for kind in ['plain','moments','mean_only','random_moments','group_mean']:
    before=torch.get_rng_state();model=C4MomentNet(copy.deepcopy(baseline),kind).eval();assert torch.equal(before,torch.get_rng_state())
    with torch.no_grad():out=model(x)
    assert torch.isfinite(out).all()
    if kind!='group_mean':assert torch.equal(out,expected)
    count=sum(p.numel() for p in model.parameters());expected_extra=18528 if kind in ['moments','mean_only','random_moments'] else 0;assert count==original+expected_extra
    flops=counted_flops(model,x,original_counter)
    budgets.append(dict(kind=kind,parameters=count,new_parameters=count-original,counted_flops224=flops,zero_init_logits_exact=kind!='group_mean',operator_path=model.operator_path))
    if kind=='moments':
        model.zero_grad(set_to_none=True);model(x).square().mean().backward()
        adapters=[p for n,p in model.named_parameters() if '.adapter.' in n];assert all(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum())>0 for p in adapters)
    del model
baseflop=budgets[0]['counted_flops224'];assert all(row['counted_flops224']==baseflop+202309632 for row in budgets[1:4])
assert budgets[4]['counted_flops224']==baseflop+86704128
report=dict(passed=True,checks=checks,finite_difference_gradcheck=True,CPU_RNG_preserved=True,misaligned_grid_rejected=True,shared_rotated_filter_gradient=True,
    integration=budgets,scope='CPU synthetic/operator and pretrained224 integration only; no CLE model training or performance evidence.',
    hypothesis='Cheap pretrained-filter orientation moments may recover some orientation-ensemble robustness; unproven efficacy/novelty.',
    theorem_scope='Only the mean/std auxiliary branch on aligned square stride4 grids is C4 spatial-equivariant; residual whole module and entire Swin are not claimed equivariant.',
    source_sha256={n:sha(S/n) for n in ['c4_moment_stem.py','check_c4_moment_stem.py','C4_MOMENT_PROTOTYPE_PROTOCOL_ZH.md']})
(S/'numeric_preflight_cpu.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(dict(passed=True,integration=budgets),ensure_ascii=False),flush=True)
