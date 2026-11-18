from pathlib import Path
import ast,gc,hashlib,json,os,sys,types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
S=Path(__file__).resolve().parent;R=S.parents[1];Q=R/'experiments/ac_qualification_20261024_window01';sys.path[:0]=[str(S),str(Q),str(R/'experiments'),str(R)]
import run_qualification as q
from qualification_model import QualifiedBackbone,counted_flops as base_flops
from c4_moment_stem import C4MomentStem,counted_flops as moment_flops
from harmonic_partition import descriptor,power_partition,lifted_responses,make_partition_stem
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available();os.nice(10);torch.set_num_threads(2);q.verify();rng=np.random.default_rng(20261025);r=rng.normal(size=(3,4,2,3,3));x=torch.tensor(r,dtype=torch.double,requires_grad=True)
 d,e1,e2=power_partition(x);fft=np.fft.fft(r,axis=1)/4;e1n=2*np.abs(fft[:,1])**2;e2n=np.abs(fft[:,2])**2;var=r.var(axis=1);assert np.max(np.abs(e1.detach().numpy()-e1n))<1e-14 and np.max(np.abs(e2.detach().numpy()-e2n))<1e-14 and np.max(np.abs(var-e1n-e2n))<1e-14;assert np.max(np.abs(d.detach().numpy()-(e2n-e1n)/(var+1e-6)))<1e-14
 for k in range(4):assert torch.allclose(d,descriptor(torch.roll(x,k,1),'shape'),atol=1e-14,rtol=1e-14)
 assert not torch.allclose(descriptor(x,'permuted'),descriptor(torch.roll(x,1,1),'permuted'),atol=1e-6)
 u=torch.tensor([1.,0.,-1.,0.],dtype=torch.double).reshape(1,4,1,1,1);v=torch.tensor([1.,-1.,1.,-1.],dtype=torch.double).reshape_as(u)/np.sqrt(2);assert torch.allclose(u.mean(1),v.mean(1),atol=1e-14) and torch.allclose(u.var(1,unbiased=False),v.var(1,unbiased=False),atol=1e-14);assert float(abs(descriptor(u,'shape')-descriptor(v,'shape')))>1.99
 assert torch.autograd.gradcheck(lambda z:descriptor(z,'shape'),(x,),eps=1e-6,atol=1e-5,rtol=1e-4)
 for scale in [0.,1e-9,1.]:
  z=(torch.ones(2,4,1,2,2,dtype=torch.double)*scale).requires_grad_()
  for kind in ['shape','permuted','duplicate_std','raw_h2']:
   y=descriptor(z,kind);grad=torch.autograd.grad(y.sum(),z,retain_graph=True)[0];assert torch.isfinite(y).all() and torch.isfinite(grad).all()
 gain=3.;actual=descriptor(gain*x,'shape')-d;expected=(e2-e1)*(gain**2/(gain**2*(e1+e2)+1e-6)-1/(e1+e2+1e-6));assert torch.allclose(actual,expected,atol=1e-14,rtol=1e-10)
 # Execute only the inspected official pure magnitude function, with no TensorFlow installation/import.
 tree=ast.parse((S/'upstream/official_hnet_lite.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stack_magnitudes')
 assert all(not isinstance(n,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal)) for n in ast.walk(fn));assert all(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='tf' and n.func.attr in ['reduce_sum','square','sqrt','maximum'] for n in ast.walk(fn) if isinstance(n,ast.Call))
 shim=types.SimpleNamespace(square=lambda a:a.square() if isinstance(a,torch.Tensor) else np.square(a),reduce_sum=lambda a,axis,keep_dims:a.sum(tuple(axis),keepdim=keep_dims) if isinstance(a,torch.Tensor) else np.sum(a,axis=tuple(axis),keepdims=keep_dims),sqrt=lambda a:a.sqrt() if isinstance(a,torch.Tensor) else np.sqrt(a),maximum=lambda a,b:torch.maximum(a,torch.as_tensor(b,dtype=a.dtype,device=a.device)) if isinstance(a,torch.Tensor) else np.maximum(a,b))
 scope={'tf':shim,'__builtins__':{}};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<whitelisted official stack_magnitudes>','exec'),scope);official=scope['stack_magnitudes'];fx=torch.fft.fft(x,dim=1,norm='forward')[:,2];complex6=torch.stack([fx.real,fx.imag],dim=1).permute(0,3,4,1,2).unsqueeze(3);ref=official(complex6)[:,:,:,0,0,:].permute(0,3,1,2);ours=descriptor(x,'raw_h2');assert torch.allclose(ref,ours,atol=1e-14,rtol=1e-14);g1=torch.autograd.grad(ref.sum(),x,retain_graph=True)[0];g2=torch.autograd.grad(ours.sum(),x,retain_graph=True)[0];assert torch.allclose(g1,g2,atol=1e-14,rtol=1e-14)
 f2=np.stack([fft[:,2].real,fft[:,2].imag],axis=1).transpose(0,3,4,1,2)[:,:,:,None,:,:];nr=official(f2)[:,:,:,0,0,:].transpose(0,3,1,2);assert np.max(np.abs(nr-ours.detach().numpy()))<1e-14
 torch.manual_seed(20261025);base=nn.Conv2d(3,4,4,stride=4).double();image=torch.randn(2,3,12,12,dtype=torch.double);orbit=lifted_responses(base,image,base(image));rot=torch.rot90(image,1,(-2,-1));rotorbit=lifted_responses(base,rot,base(rot))
 for kind in ['shape','duplicate_std','raw_h2']:assert torch.allclose(descriptor(rotorbit,kind),torch.rot90(descriptor(orbit,kind),1,(-2,-1)),atol=1e-11,rtol=1e-11),kind
 assert not torch.allclose(descriptor(rotorbit,'permuted'),torch.rot90(descriptor(orbit,'permuted'),1,(-2,-1)),atol=1e-6)
 cls=make_partition_stem(C4MomentStem);small=cls(nn.Conv2d(3,4,4,stride=4),'shape');opt=torch.optim.AdamW(small.parameters(),lr=1e-3);target=torch.randn(2,4,3,3)
 for _ in range(2):opt.zero_grad();loss=F.mse_loss(small(image.float()),target);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in small.parameters());opt.step()
 del small,opt;base_image=torch.randn(3,3,224,224);initial=None;budgets=[]
 for kind in ['shape','permuted','duplicate_std','raw_h2']:
  q.core.set_seed(45);model=QualifiedBackbone('convnext_tiny',Q/'pretrained').eval();old={id(p):p.detach().clone() for p in model.parameters()};state=torch.get_rng_state().clone()
  with torch.no_grad():expected=model(base_image)
  model.encoder[0][0][0]=cls(model.encoder[0][0][0],kind);assert torch.equal(state,torch.get_rng_state()) and all(torch.equal(p,old[id(p)]) for p in model.parameters() if id(p) in old)
  with torch.no_grad():actual=model(base_image)
  assert torch.equal(actual,expected)
  stem=model.encoder[0][0][0];raw=stem.base(base_image);mean,std=stem.statistics(base_image,raw);group=lifted_responses(stem.base,base_image,raw);assert torch.equal(mean,group.mean(1)) and torch.equal(std,((group-group.mean(1,keepdim=True)).square().mean(1)+1e-6).sqrt())
  count=sum(p.numel() for p in model.parameters());assert count==27850275;flops=moment_flops(model,base_image[:1],base_flops);assert flops==9169643520
  model.train();F.cross_entropy(model(base_image),torch.tensor([0,1,2])).backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());norm=float(stem.partition_adapter.weight.grad.norm());assert norm>0;budgets.append(dict(kind=kind,parameters=count,extra_over_Mom=9312,counted_inference_flops=flops,initial_logits_exact=True,original_parameter_ids_and_values_preserved=True,RNG_preserved=True,new_projection_gradient_norm=norm));del model,old;gc.collect()
 proof=dict(passed=True,NumPy_DFT_and_Parseval=True,same_mean_std_different_power_shape=True,C4_cyclic_shift_invariant=True,noncyclic_control_breaks_group_relation=True,input90_spatial_covariance=True,constant_low_energy_finite=True,scale_eps_deviation_formula_checked=True,double_gradcheck=True,official_HNet_magnitude_output_and_gradient=True,official_function_AST_pure_allowlist=True,public_Tiny224_logits_RNG_parameters_and_finite_backward=True,all4_equal_budgets=budgets,two_synthetic_stem_updates_finite=True,CLE_images=0,CLE_training=0,test_predictions=0,whole_network_equivariance_claim=False,overall_research_complete=False)
 (S/'numeric_preflight_cpu.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
