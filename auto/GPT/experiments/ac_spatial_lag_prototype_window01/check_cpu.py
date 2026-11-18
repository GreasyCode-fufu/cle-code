from pathlib import Path
import copy,hashlib,itertools,json,math,os,sys,traceback
import numpy as np
import torch
from torch.nn import functional as F
S=Path(__file__).resolve().parent;R=S.parents[1]
sys.path[:0]=[str(R),str(R/'experiments/ac_qualification_20261024_window01'),str(R/'experiments/ac_structure_20261024_window01'),str(R/'experiments')]
from models.optimize.ac_spatial_lag_prototype_window01.lag_correlation import lag_descriptor,LagCorrelationNet,ARMS
from models.optimize.ac_spatial_lag_prototype_window01.c4_moment_stem import counted_flops as moment_flops
from qualification_model import QualifiedBackbone,counted_flops
import run_qualification as q
torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
save=lambda n,d:(S/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def oracle(a,kind,perm=None,eps=1e-6):
    if kind=='permuted':a=a.reshape(*a.shape[:2],-1)[...,perm].reshape(a.shape)
    c=a-a.mean((-2,-1),keepdims=True);b,ch,h,w=a.shape;values=[];fft_error=0.
    fft=np.fft.fft2(c,s=(2*h-1,2*w-1),axes=(-2,-1))
    linear=np.fft.ifft2(np.conj(fft)*fft,axes=(-2,-1)).real
    for lag in ([1,1] if kind=='duplicate' else [1,3]):
        acc=[]
        for axis in [1,0]:
            if (w if axis else h)<=lag:continue
            u=c[...,:-lag] if axis else c[...,:-lag,:]
            v=c[...,lag:] if axis else c[...,lag:,:]
            product=np.zeros((b,ch));e1=np.zeros((b,ch));e2=np.zeros((b,ch))
            for row in range(u.shape[-2]):
                for col in range(u.shape[-1]):
                    product+=u[...,row,col]*v[...,row,col];e1+=u[...,row,col]**2;e2+=v[...,row,col]**2
            fft_sum=linear[...,0,lag] if axis else linear[...,lag,0]
            fft_error=max(fft_error,float(np.max(np.abs(product-fft_sum))))
            if kind=='acf':rho=product/((c*c).sum((-2,-1))+eps*h*w)
            else:
                count=u.shape[-2]*u.shape[-1]
                rho=(product/count)/(np.sqrt(e1/count+eps)*np.sqrt(e2/count+eps))
            acc.append(rho)
        values.append(np.mean(acc,0) if acc else np.zeros((b,ch)))
    return np.concatenate(values,1),fft_error

def main():
    assert os.getuid()==1000 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cfg=json.loads((S/'config.json').read_text());assert all(sha(R/p)==h for p,h in cfg['sources'].items()) and all(sha(R/p)==h for p,h in cfg['locked'].items())
    rng=np.random.default_rng(20261025);a=rng.normal(size=(2,4,7,7));x=torch.tensor(a,dtype=torch.float64,requires_grad=True);perm=rng.permutation(49);tp=torch.tensor(perm,dtype=torch.long)
    numeric={}
    for kind in ['valid','permuted','duplicate','acf']:
        ref,fft_error=oracle(a,kind,perm);actual=lag_descriptor(x,kind,False,tp)
        error=float(np.max(np.abs(actual.detach().numpy()-ref)));assert error<1e-12 and fft_error<1e-11 and np.max(np.abs(ref))<=1+1e-12
        normalized=(ref-ref.mean(1,keepdims=True))/np.sqrt(ref.var(1,keepdims=True)+1e-5)
        assert np.max(np.abs(lag_descriptor(x,kind,True,tp).detach().numpy()-normalized))<1e-11
        assert torch.allclose(lag_descriptor(x[:1],kind,True,tp),lag_descriptor(x,kind,True,tp)[:1],atol=1e-12,rtol=0)
        small=torch.randn(1,2,7,7,dtype=torch.float64,requires_grad=True)
        assert torch.autograd.gradcheck(lambda v:lag_descriptor(v,kind,True,tp),(small,),eps=1e-6,atol=3e-5,rtol=1e-3)
        flat=torch.ones(2,4,7,7,dtype=torch.float64,requires_grad=True);v=lag_descriptor(flat,kind,True,tp);v.sum().backward();assert torch.count_nonzero(v)==0 and torch.isfinite(flat.grad).all()
        numeric[kind]=dict(numpy_max_error=error,zero_padding_FFT_numerator_max_error=fft_error,double_gradcheck=True,constant_finite=True,batch_independent=True)
    for h,w in [(1,1),(1,7),(7,1),(7,9),(3,3)]:
        v=torch.randn(2,3,h,w,dtype=torch.float64,requires_grad=True)
        for kind in ['valid','duplicate','acf']:
            y=lag_descriptor(v,kind);g=torch.autograd.grad(y.square().sum(),v,retain_graph=True)[0];assert torch.isfinite(y).all() and torch.isfinite(g).all()
            assert torch.allclose(y,lag_descriptor(torch.rot90(v,1,(-2,-1)),kind),atol=1e-10,rtol=0)
            assert torch.allclose(y,lag_descriptor(v.flip(-1),kind),atol=1e-10,rtol=0)
    assert torch.allclose(lag_descriptor(x+5),lag_descriptor(x),atol=1e-11,rtol=0)
    assert torch.allclose(lag_descriptor(-x),lag_descriptor(x),atol=1e-11,rtol=0)
    scale_error=float((lag_descriptor(2*x,'valid',False)-lag_descriptor(x,'valid',False)).abs().max().detach());assert scale_error<1e-6
    # Equal complete marginal distributions, unequal spatial arrangement.
    check=np.where(np.indices((7,7)).sum(0)%2,1.,-1.)
    block=np.sort(check.flatten()).reshape(7,7)
    assert np.array_equal(np.sort(check.ravel()),np.sort(block.ravel()))
    cv=lag_descriptor(torch.tensor(check)[None,None],'valid',False);bv=lag_descriptor(torch.tensor(block)[None,None],'valid',False)
    layout_difference=float((cv-bv).abs().max());assert layout_difference>.1
    # Exact equal lag1, different lag3 counterexample among balanced binary maps.
    seen={};counter=None
    for positives in itertools.combinations(range(16),8):
        image=-np.ones(16);image[list(positives)]=1.;image=image.reshape(4,4)
        lag1=int((image[:,:-1]*image[:,1:]).sum()+(image[:-1]*image[1:]).sum())
        lag3=int((image[:,:-3]*image[:,3:]).sum()+(image[:-3]*image[3:]).sum())
        if lag1 in seen and seen[lag1][0]!=lag3:counter=(seen[lag1][1],image);break
        seen[lag1]=(lag3,image)
    assert counter is not None
    ca=lag_descriptor(torch.tensor(counter[0])[None,None],'valid',False);cb=lag_descriptor(torch.tensor(counter[1])[None,None],'valid',False)
    assert abs(float(ca[0,0]-cb[0,0]))<1e-15 and abs(float(ca[0,1]-cb[0,1]))>.1
    one=torch.tensor([2.,3.,-1.,4.,0.,-2.,1.],dtype=torch.float64)[None,None,None,:];center=one.numpy().ravel()-float(one.mean());expected=np.dot(center[:-1],center[1:])/np.dot(center,center)
    nist_error=abs(float(lag_descriptor(one,'acf',False,eps=0)[0,0])-expected);assert nist_error<1e-15
    assert not torch.allclose(lag_descriptor(x,'valid'),lag_descriptor(x,'permuted',permutation=tp))
    stress=(torch.randn(2,4,7,7)*1e6).requires_grad_()
    for kind in ['valid','permuted','duplicate','acf']:
        y=lag_descriptor(stress,kind,permutation=tp);g=torch.autograd.grad(y.square().sum(),stress,retain_graph=True)[0];assert torch.isfinite(y).all() and torch.isfinite(g).all()
    toy={}
    for kind in ['valid','permuted','duplicate','acf']:
        z=lag_descriptor(x.detach().float(),kind,permutation=tp);p=torch.nn.Linear(8,3);torch.nn.init.zeros_(p.weight);torch.nn.init.zeros_(p.bias);opt=torch.optim.SGD(p.parameters(),lr=.01);losses=[]
        for _ in range(2):
            opt.zero_grad();loss=F.cross_entropy(p(z),torch.tensor([0,1]));loss.backward();assert torch.isfinite(loss) and p.weight.grad.norm()>0;losses.append(float(loss.detach()));opt.step()
        assert losses[1]<losses[0];toy[kind]=losses
    init=[];q.verify()
    for seed in [45,46,47]:
        asset=q.core.read_json(q.STUDY/'pretrained_manifest.json')['assets']['convnext_tiny'];assert sha(R/asset['path'])==asset['sha256']
        q.core.set_seed(seed);base=QualifiedBackbone('convnext_tiny',q.STUDY/'pretrained').eval();image=torch.randn(1,3,224,224)
        with torch.no_grad():expected=base(image)
        for arm in ['plain','mom_only',*ARMS]:
            old=copy.deepcopy(base);ids={id(p) for p in old.parameters()};state=torch.random.get_rng_state().clone();net=LagCorrelationNet(old,arm).eval()
            assert torch.equal(state,torch.random.get_rng_state()) and ids<=set(id(p) for p in net.parameters()) and net.original_parameter_ids==ids
            with torch.no_grad():actual=net(image)
            assert torch.equal(actual,expected)
            count=sum(p.numel() for p in net.parameters());assert count==27822435+(18528 if arm.startswith('mom_') else 0)+(4611 if arm in ARMS else 0)
            grad=None
            if seed==45 and arm in ARMS:
                net.train();net.zero_grad(set_to_none=True);F.cross_entropy(net(image),torch.tensor([1])).backward();grad=float(net.projection.weight.grad.norm());assert grad>0 and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            budget=moment_flops(net,image,counted_flops) if seed==45 else None
            init.append(dict(seed=seed,arm=arm,public_logit_max_error=0.,rng_preserved=True,original_parameter_objects_retained=True,parameters=count,counted_flops=budget,projection_first_gradient_norm=grad));del net,old
        del base
    result=dict(passed=True,numeric=numeric,rectangular_degenerate_reflection_rotation_checks=True,equal_marginals_layout_difference=layout_difference,equal_lag1_distinct_lag3=dict(first=ca.tolist(),second=cb.tolist()),mature_NIST_1D_noeps_max_error=nist_error,scale_approx_error=scale_error,toy_two_updates=toy,public_initialization=init,CLE_images=0,CLE_training=0,val_images=0,test_images=0,source_sha256=cfg['sources'],innovation_confirmed=False,overall_research_complete=False)
    save('numeric_preflight_cpu.json',result);save('complete.json',dict(passed=True,proof_sha256=sha(S/'numeric_preflight_cpu.json'),overall_research_complete=False));print(json.dumps(dict(passed=True,public_variants=len(init),proof_sha256=sha(S/'numeric_preflight_cpu.json'))),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:save('failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
