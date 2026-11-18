from pathlib import Path
import hashlib,json,sys
import torch
S=Path(__file__).resolve().parent;R=S.parents[1];Q=R/'experiments/ac_qualification_20261024_window01'
sys.path[:0]=[str(S),str(Q)]
from c4_moment_stem import C4MomentNet,counted_flops
from qualification_model import QualifiedBackbone,counted_flops as base_counter
torch.set_num_threads(1);torch.manual_seed(45)
base=QualifiedBackbone('swin_t',Q/'pretrained').eval();x=torch.randn(1,3,384,384)
with torch.no_grad():expected=base(x)
model=C4MomentNet(base,'moments').eval()
with torch.no_grad():actual=model(x)
assert torch.equal(actual,expected)
flops=counted_flops(model,x,base_counter);reference=json.loads((Q/'preflight.json').read_text());plain=next(b['counted_flops'] for b in reference['budgets'] if b['arm']=='swin_t_B')
assert flops-plain==594542592 and sum(p.numel() for p in model.parameters())==27540189
model(x).square().mean().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
assert any(p.grad is not None and float(p.grad.abs().sum())>0 for n,p in model.named_parameters() if '.adapter.' in n)
result=dict(passed=True,device='CPU',crop384_synthetic_zero_init_logits_exact=True,finite_backward=True,parameters=27540189,new_parameters=18528,
    counted_flops=flops,plain_counted_flops=plain,additional_flops=flops-plain,relative_flop_increase=flops/plain-1,
    cle_images=0,network_training=0,GPU_preflight=False,source_sha256={n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in ['c4_moment_stem.py','check_c4_moment_384.py']})
(S/'numeric_preflight_cpu384.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
