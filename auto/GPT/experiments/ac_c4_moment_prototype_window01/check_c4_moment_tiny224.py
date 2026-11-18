from pathlib import Path
import copy,hashlib,json,sys
import torch
S=Path(__file__).resolve().parent;R=S.parents[1];Q=R/'experiments/ac_qualification_20261024_window01'
sys.path[:0]=[str(S),str(Q)]
from c4_moment_stem import C4MomentNet,counted_flops
from qualification_model import QualifiedBackbone,counted_flops as counter
torch.set_num_threads(1);torch.manual_seed(45)
base=QualifiedBackbone('convnext_tiny',Q/'pretrained').eval();x=torch.randn(1,3,224,224)
with torch.no_grad():target=base(x)
budgets=[]
for kind in ['plain','moments','mean_only','random_moments','group_mean']:
    before=torch.get_rng_state();model=C4MomentNet(copy.deepcopy(base),kind).eval();assert torch.equal(before,torch.get_rng_state())
    with torch.no_grad():actual=model(x)
    assert torch.isfinite(actual).all()
    if kind!='group_mean':assert torch.equal(actual,target)
    flops=counted_flops(model,x,counter);parameters=sum(p.numel() for p in model.parameters())
    assert parameters==27822435+(18528 if kind in ['moments','mean_only','random_moments'] else 0)
    if kind in ['moments','mean_only','random_moments']:assert flops==9111840768
    if kind=='moments':
        model(x).square().mean().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(p.grad is not None and float(p.grad.abs().sum())>0 for name,p in model.named_parameters() if '.adapter.' in name)
    budgets.append(dict(kind=kind,parameters=parameters,counted_flops=flops,pretrained_zero_init_logits_exact=kind!='group_mean'))
result=dict(passed=True,device='CPU',baseline='frozen B0 convnext_tiny_A',budget=budgets,new_parameters=18528,cle_images=0,network_training=0,GPU_preflight=False,
    scope='Synthetic224 integration in the already-frozen primary backbone; not selection of a weaker baseline or efficacy evidence.',
    source_sha256={n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in ['c4_moment_stem.py','check_c4_moment_tiny224.py']})
(S/'numeric_preflight_cpu_tiny224.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
