"""Synthetic CPU integration audit; no val scores and no CLE training."""
from pathlib import Path
import collections,copy,hashlib,json,sys
import torch
from torch import nn
S=Path(__file__).resolve().parent;R=S.parents[1];Q=R/'experiments/ac_qualification_20261024_window01'
sys.path[:0]=[str(S),str(Q)]
from field_model import FieldModel,FieldConv,FieldAttention,FieldPool,counted_flops
from field_ops_v2 import unit_circle
from qualification_model import QualifiedBackbone,counted_flops as original_flops
torch.set_num_threads(1);torch.manual_seed(20261024)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert json.loads((S/'numeric_preflight_v2.json').read_text())['passed']
budgets=json.loads((Q/'preflight.json').read_text())['budgets']
results=[]
for family in ['convnext_tiny','efficientnet_v2_s','swin_t','convnext_base']:
 torch.manual_seed(45);baseline=QualifiedBackbone(family,Q/'pretrained').eval()
 original_parameters=sum(p.numel() for p in baseline.parameters())
 gray=torch.rand(1,1,224,224);rgb=gray.expand(-1,3,-1,-1).clone();mask=unit_circle(224,224)
 x=torch.cat((rgb,mask),1)
 with torch.no_grad():expected=baseline(rgb)
 before={k:v.detach().clone() for k,v in baseline.state_dict().items()}
 for kind in ['plain','fov_residual','no_field','no_local_norm','no_pool','field_pool','reference']:
  model=FieldModel(copy.deepcopy(baseline),family,kind).eval()
  # Every public source/head tensor is preserved under the wrapper's explicit key mapping.
  used=[]
  for name,value in model.backbone.state_dict().items():
   source_name=name.replace('.base.','.')
   if source_name in before:
    assert torch.equal(value,before[source_name]);used.append(source_name)
   else:assert name.endswith('.alpha'),name
  assert set(used)==set(before) and len(used)==len(before)
  with torch.no_grad():output=model(x)
  assert torch.isfinite(output).all() and model.context.mask is None and not model.context.cache
  exact=kind in ['plain','fov_residual','no_field','no_local_norm','no_pool']
  if exact:assert torch.equal(output,expected),(family,kind,float((output-expected).abs().max()))
  n=sum(p.numel() for p in model.parameters());gates=[(name,p) for name,p in model.named_parameters() if name.endswith('.alpha')]
  assert n==original_parameters+len(gates)
  ids=[id(p) for p in model.parameters()];assert len(ids)==len(set(ids))
  # EfficientNet's SE pools must all retain their original class and count.
  se_count=sum(isinstance(m,nn.AdaptiveAvgPool2d) for m in baseline.encoder[0].modules())
  se_model_count=sum(isinstance(m,nn.AdaptiveAvgPool2d) for m in model.encoder[0].modules())
  assert se_count==se_model_count
  flop=counted_flops(model,x)
  if kind=='plain':assert flop==next(b['counted_flops'] for b in budgets if b['arm']==family+'_A')
  record=dict(family=family,kind=kind,parameters=n,new_gate_parameters=len(gates),counted_flops=flop,
   zero_init_baseline_exact=exact,wrapped_operator_types=dict(collections.Counter(p['type'] for p in model.operator_paths)),
   inner_se_pools_preserved=se_count,context_cleared=True)
  if kind=='fov_residual':
   model.zero_grad(set_to_none=True);pred=model(x);pred.square().mean().backward()
   assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
   active=sum(p.grad is not None and float(p.grad.abs())>0 for _,p in gates)
   assert active>0
   record.update(finite_backward=True,nonzero_gate_gradients=int(active),no_optimizer_update=True)
   assert all(torch.equal(value.detach(),before[name.replace('.base.','.')]) for name,value in model.backbone.state_dict().items() if name.replace('.base.','.') in before)
   del pred
  results.append(record);del model,output
 del baseline,before,expected,rgb,gray,x,mask
report=dict(passed=True,scope='Four-family synthetic224 CPU architecture integration only. No CLE fitting, no val metrics, no baseline selection, no test access.',
 rows=results,source_sha256={n:sha(S/n) for n in ['field_model.py','check_field_model_cpu.py','field_ops_v2.py','FIELD_INTEGRATION_PROTOCOL.md']},
 next_required='Wait final qualification audit and baseline freeze; then full selected-backbone actual-image CUDA AMP and original427 val equivalence before preregistered structural training.')
(S/'model_integration_cpu.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(passed=True,integrated_backbones=4,variants=7,cle_training=0,test_access=0),ensure_ascii=False))
