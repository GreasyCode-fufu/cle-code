from pathlib import Path
import ast,copy,hashlib,json,os,sys,traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
S=Path(__file__).resolve().parent;R=S.parents[1]
sys.path[:0]=[str(R),str(R/'experiments/ac_qualification_20261024_window01'),str(R/'experiments/ac_structure_20261024_window01'),str(R/'experiments')]
from models.optimize.ac_salience_contrast_prototype_window01.salience_contrast import SalienceReadout,SalienceNet,ARMS,counted_flops
from qualification_model import QualifiedBackbone,counted_flops as base_flops
import run_qualification as q
torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
save=lambda n,v:(S/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def np_ln(a):return (a-a.mean(-1,keepdims=True))/np.sqrt(a.var(-1,keepdims=True)+1e-5)
def np_oracle(net,x):
 h=x.reshape(*x.shape[:2],-1).transpose(0,2,1);z=np_ln(h)
 v=np.tanh(z@net.v.weight.detach().numpy().T+net.v.bias.detach().numpy())
 if net.u is not None:v*=1/(1+np.exp(-(z@net.u.weight.detach().numpy().T+net.u.bias.detach().numpy())))
 score=(v@net.w.weight.detach().numpy().T)[...,0];a=np.exp(score-score.max(1,keepdims=True));a/=a.sum(1,keepdims=True)
 if net.kind=='misaligned':a=a[:,net.permutation.numpy()]
 value=h if net.kind=='weightedmean' else h-h.mean(1,keepdims=True)
 d=np.stack([sum(a[b,k]*value[b,k] for k in range(value.shape[1])) for b in range(value.shape[0])])
 return d,a

def official():
 src=S/'references/official_model.py';license=S/'references/LICENSE_AttentionDeepMIL'
 for p,h in [(src,'e99adc74d63240c637f3a1ec3174c4a2d9f1aa26'),(license,'58ed84464a359db312c3da36f040bfeba8f8572d')]:
  b=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==h
 assert 'MIT License' in license.read_text() and 'Maximilian Ilse and Jakub Tomczak' in license.read_text()
 cls=next(n for n in ast.parse(src.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='GatedAttention')
 forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
 selected=[n for n in forward.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['A_V','A_U','A','Z']]
 assert len(selected)==6
 assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.Name) and n.id in ['open','exec','eval','compile','__import__','os','sys','subprocess'] for root in selected for n in ast.walk(root))
 args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self'),ast.arg(arg='H')],kwonlyargs=[],kw_defaults=[],defaults=[])
 method=ast.FunctionDef(name='pool',args=args,body=selected+[ast.Return(value=ast.Tuple(elts=[ast.Name(id='Z',ctx=ast.Load()),ast.Name(id='A',ctx=ast.Load())],ctx=ast.Load()))],decorator_list=[])
 env={'torch':torch,'F':F};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(src),'exec'),env)
 net=SalienceReadout('contrast',channels=4,rank=3).double();obj=type('OperatorOnly',(),{})();obj.attention_V=nn.Sequential(net.v,nn.Tanh());obj.attention_U=nn.Sequential(net.u,nn.Sigmoid());obj.attention_w=net.w
 h=torch.randn(9,4,dtype=torch.float64,requires_grad=True);z,a=env['pool'](obj,h);aa=net.weights(h[None]);zz=aa@h;err=float((z-zz).abs().max().detach());probe=torch.randn_like(z)
 gg=torch.autograd.grad((z*probe).sum(),[h,*net.v.parameters(),*net.u.parameters(),*net.w.parameters()],retain_graph=True)
 g=torch.autograd.grad((zz*probe).sum(),[h,*net.v.parameters(),*net.u.parameters(),*net.w.parameters()]);ge=max(float((x-y).abs().max()) for x,y in zip(gg,g));assert err<1e-12 and ge<1e-12 and torch.allclose(a,aa,atol=1e-12,rtol=0)
 return dict(passed=True,source_sha256=sha(src),license_sha256=sha(license),restricted_AST_assignment_targets=['A_V','A_U','A','Z'],forward_error=err,all_input_and_score_parameter_gradient_error=ge,full_paper_reproduction=False)

def main():
 assert os.getuid()==1000 and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
 cfg=json.loads((S/'config.json').read_text());assert all(sha(R/p)==h for name in ['sources','locked'] for p,h in cfg[name].items())
 torch.manual_seed(20261025);rng=np.random.default_rng(20261025);x=torch.randn(2,4,7,7,dtype=torch.float64,requires_grad=True);proof={}
 for kind in ['contrast','weightedmean','misaligned','ungated']:
  net=SalienceReadout(kind,channels=4,rank=3).double();d,a=np_oracle(net,x.detach().numpy());out=net.descriptor(x,False);err=float(np.abs(out.detach().numpy()-d).max());assert err<1e-12
  assert np.abs(net.descriptor(x).detach().numpy()-np_ln(d)).max()<1e-10
  assert torch.autograd.gradcheck(lambda v:net.descriptor(v),(x[:1],),eps=1e-6,atol=3e-5,rtol=1e-3)
  assert torch.allclose(net.descriptor(x[:1]),net.descriptor(x)[:1],atol=1e-11,rtol=0)
  flat=torch.ones(2,4,7,7,dtype=torch.float64,requires_grad=True);o=net.descriptor(flat);g=torch.autograd.grad(o.square().sum(),flat)[0];assert torch.isfinite(o).all() and torch.isfinite(g).all()
  if kind!='weightedmean':assert o.abs().max()<1e-10
  if kind!='misaligned':
   perm=torch.tensor(rng.permutation(49));xp=x.flatten(2)[:,:,perm].reshape_as(x);assert torch.allclose(net.descriptor(x),net.descriptor(xp),atol=1e-10,rtol=0)
  proof[kind]=dict(numpy_error=err,doublegradcheck=True,batch_independent=True,finite_flat=True,permutation_invariant=kind!='misaligned')
 for h,w in [(1,1),(1,7),(7,1),(7,9)]:
  v=torch.randn(2,4,h,w,dtype=torch.float64,requires_grad=True)
  for kind in ['contrast','weightedmean','ungated']:
   net=SalienceReadout(kind,channels=4,rank=3).double();d=net.descriptor(v);g=torch.autograd.grad(d.square().sum(),v,retain_graph=True)[0];assert torch.isfinite(d).all() and torch.isfinite(g).all()
 # Same pooled mean with opposite skew: linear score can select different tails.
 v1=torch.tensor([[-2.,0.,2.],[-1.,-1.,2.],[1.,-2.,1.],[3.,-1.,-2.]],dtype=torch.float64);v2=v1.clone();v2[1]=torch.tensor([-2.,1.,1.],dtype=torch.float64)
 assert torch.equal(v1.mean(-1),v2.mean(-1));counter=SalienceReadout('contrast',channels=4,rank=3).double()
 d1=counter.descriptor(v1[None,:,:,None],False);d2=counter.descriptor(v2[None,:,:,None],False);assert not torch.allclose(d1,d2,atol=1e-6,rtol=0)
 official_proof=official();toy={}
 for kind in ['contrast','weightedmean','misaligned','ungated']:
  net=SalienceReadout(kind);v=torch.randn(3,768,7,7);opt=torch.optim.SGD(net.parameters(),lr=.001);records=[]
  for step in range(2):
   opt.zero_grad();loss=F.cross_entropy(net(v),torch.tensor([0,1,2]));loss.backward();head=float(net.classifier.weight.grad.norm());vg=float(net.v.weight.grad.norm());wg=float(net.w.weight.grad.norm());ug=float(net.u.weight.grad.norm()) if net.u is not None else None
   assert head>0 and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters());assert (vg==wg==0 and ug in [None,0.]) if step==0 else (vg>0 and wg>0 and (ug is None or ug>0))
   records.append(dict(CE=float(loss.detach()),classifier_grad=head,v_grad=vg,u_grad=ug,w_grad=wg));opt.step()
  assert records[1]['CE']<records[0]['CE'];toy[kind]=records
 init=[];q.verify()
 for seed in [45,46,47]:
  q.core.set_seed(seed);base=QualifiedBackbone('convnext_tiny',q.STUDY/'pretrained').eval();image=torch.randn(1,3,224,224)
  with torch.no_grad():expected=base(image)
  for arm in ['plain','mom_only',*ARMS]:
   old=copy.deepcopy(base);ids={id(p) for p in old.parameters()};state=torch.get_rng_state().clone();net=SalienceNet(old,arm).eval();assert torch.equal(state,torch.get_rng_state()) and ids<={id(p) for p in net.parameters()}
   out=net(image);assert torch.equal(out,expected);F.cross_entropy(out,torch.tensor([1])).backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
   extra=14627 if arm=='mom_ungated' else (26931 if arm in ARMS else 0);params=27822435+(18528 if arm.startswith('mom_') else 0)+extra;assert sum(p.numel() for p in net.parameters())==params
   flops=counted_flops(net,image,base_flops);expected_cost=8909531136+(202309632 if arm.startswith('mom_') else 0)+(1285664 if arm=='mom_ungated' else (2489888 if arm in ARMS else 0));assert flops==expected_cost,(arm,flops,expected_cost)
   if arm in ARMS:assert net.readout.classifier.weight.grad.norm()>0 and net.readout.v.weight.grad.norm()==0 and net.readout.w.weight.grad.norm()==0
   if arm=='salience_only':assert not hasattr(net.encoder[0][0][0],'adapter')
   init.append(dict(seed=seed,arm=arm,parameters=params,counted_flops=flops,logit_exact=True,RNG_preserved=True,original_parameter_ids_preserved=True,finite_backward=True));del net,old
  del base
 save('numeric_preflight_cpu.json',dict(passed=True,numeric=proof,official=official_proof,same_mean_different_content_counterexample=True,toy_updates=toy,public_initializations=init,CLE_images=0,new_training=0,test_images=0,logical_FLOPs_exclusions='LN/softmax/tanh/sigmoid/centering/elementwise/memory; attention weighted pooling BMM included',novelty_confirmed=False))
 save('complete.json',dict(passed=True,proof_sha256=sha(S/'numeric_preflight_cpu.json'),source_config_sha256=sha(S/'config.json'),overall_research_complete=False))
if __name__=='__main__':
 try:main()
 except Exception as e:save('failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
