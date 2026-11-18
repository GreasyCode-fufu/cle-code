from pathlib import Path
import ast,functools,hashlib,json,operator,os,sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
S=Path(__file__).resolve().parent;R=S.parents[1];sys.path.insert(0,str(S))
from crossscale_readout import bounded_cross_correlation,CrossScaleReadout,CrossScaleNet
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    rev=json.loads((S/'cpu_revision_v2.json').read_text());assert rev['original_config_sha256']==sha(S/'config.json') and all(sha(S/p)==h for p,h in rev['sources'].items());cfg=json.loads((S/'config.json').read_text());assert all(sha(S/p)==h for p,h in cfg['sources'].items());assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available();os.nice(10);torch.set_num_threads(2)
    ns={'torch':torch,'nn':nn,'functools':functools,'operator':operator,'Function':Function}
    for file,cls in [('BCNN.py','TensorProduct'),('MPNCOV.py','Covpool')]:
        code=(S/'upstream'/file).read_text();node=next(n for n in ast.parse(code).body if isinstance(n,ast.ClassDef) and n.name==cls);exec(compile(ast.Module(body=[node],type_ignores=[]),str(S/'upstream'/file),'exec'),ns)
    torch.manual_seed(20261025);a=torch.randn(2,3,2,3,dtype=torch.double,requires_grad=True);b=torch.randn(2,5,2,3,dtype=torch.double,requires_grad=True)
    original=ns['TensorProduct']([3,5])(a,b);ours=a.flatten(2).bmm(b.flatten(2).transpose(1,2))/6;assert torch.equal(original,ours)
    go=torch.autograd.grad(original.square().sum(),(a,b),retain_graph=True);gn=torch.autograd.grad(ours.square().sum(),(a,b));assert all(torch.equal(x,y) for x,y in zip(go,gn))
    cov_errors=[]
    for h in [2,7]:
        x=torch.randn(2,3,h,h,dtype=torch.double,requires_grad=True);c=ns['Covpool'].apply(x);z=x.flatten(2)-x.flatten(2).mean(-1,keepdim=True);expected=z.bmm(z.transpose(1,2))/(h*h);error=float((c-expected).abs().max().detach());assert error<(1e-12 if h==2 else 1e-6);cov_errors.append(dict(spatial=h*h,max_error=error,default_fp32_centering_coefficients_in_original=True))
    a=torch.randn(2,3,9,dtype=torch.double,requires_grad=True);b=torch.randn(2,4,9,dtype=torch.double,requires_grad=True);actual=bounded_cross_correlation(a,b)
    aa=a.detach().numpy();bb=b.detach().numpy();aa=aa-aa.mean(-1,keepdims=True);bb=bb-bb.mean(-1,keepdims=True);aa/=np.sqrt((aa*aa).mean(-1,keepdims=True)+1e-4);bb/=np.sqrt((bb*bb).mean(-1,keepdims=True)+1e-4);mat=np.einsum('bcn,bdn->bcd',aa,bb)/9;flat=mat.reshape(2,-1);oracle=flat/np.sqrt(1+(flat*flat).sum(1,keepdims=True));assert np.max(np.abs(actual.detach().numpy()-oracle))<1e-12
    perm=torch.tensor([3,0,7,2,8,6,1,5,4]);assert torch.allclose(actual,bounded_cross_correlation(a[:,:,perm],b[:,:,perm]),atol=1e-12)
    assert not torch.allclose(actual,bounded_cross_correlation(a,b[:,:,perm]),atol=1e-6)
    assert torch.allclose(actual,bounded_cross_correlation(a+2,b-7),atol=1e-12) and (actual.norm(dim=1)<1).all()
    small=(torch.randn(1,2,4,dtype=torch.double,requires_grad=True),torch.randn(1,3,4,dtype=torch.double,requires_grad=True));assert torch.autograd.gradcheck(bounded_cross_correlation,small,eps=1e-6,atol=1e-4)
    const=(torch.ones(2,3,9,dtype=torch.double,requires_grad=True),torch.ones(2,4,9,dtype=torch.double,requires_grad=True));v=bounded_cross_correlation(*const);assert torch.equal(v,torch.zeros_like(v));assert all(torch.isfinite(g).all() for g in torch.autograd.grad(v.sum(),const))
    model=CrossScaleReadout();assert sum(p.numel() for p in model.parameters())==19203;opt=torch.optim.SGD(model.parameters(),lr=.01);x3=torch.randn(4,384,14,14);x4=torch.randn(4,768,7,7)
    assert torch.equal(model(x3,x4),torch.zeros(4,3))
    for step in range(2):
        opt.zero_grad();loss=nn.functional.cross_entropy(model(x3,x4),torch.tensor([0,1,2,1]));loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
    assert model.low.weight.grad.abs().sum()>0
    sys.path[:0]=[str(R),str(R/'experiments/ac_qualification_20261024_window01'),str(R/'experiments')]
    from qualification_model import QualifiedBackbone,counted_flops
    from models.optimize.ac_moment_refinement_window01.library_cle import LibraryNet
    import run_data_study as core
    core.set_seed(45);base=LibraryNet(QualifiedBackbone('convnext_tiny',R/'experiments/ac_qualification_20261024_window01/pretrained'),'moments').eval();image=torch.randn(1,3,224,224)
    with torch.no_grad():expected=base(image)
    state=torch.get_rng_state().clone();full=CrossScaleNet(base.backbone,base.original_parameter_ids).eval();assert torch.equal(state,torch.get_rng_state())
    with torch.no_grad():assert torch.equal(full(image),expected)
    assert sum(p.numel() for p in full.parameters())==27860166
    audit=dict(passed=True,checker_revision=2,cpu_revision_sha256=sha(S/'cpu_revision_v2.json'),original_bilinear_forward_and_gradients_exact=True,covariance_parity=cov_errors,numpy_correlation_oracle=True,paired_permutation_invariance=True,unpaired_permutation_changes_response=True,additive_feature_shift_invariance=True,constant_feature_gradient_finite=True,double_gradcheck=True,readout_parameters=19203,public_moment_model_parameters=27860166,zero_initial_logits_exact=True,synthetic_optimizer_steps=2,CLE_images=0,test_predictions=0,real_neural_training=0,source_sha256=cfg['sources'],scope='CPU prototype only; effectiveness and novelty unproven')
    (S/'numeric_preflight_cpu.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:audit[k] for k in ['passed','readout_parameters','public_moment_model_parameters','CLE_images']}))
if __name__=='__main__':main()
