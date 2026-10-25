from pathlib import Path
import copy, hashlib, json, math, os, sys, traceback
import numpy as np
import torch
from torch.nn import functional as F

S=Path(__file__).resolve().parent;R=S.parents[1]
sys.path[:0]=[str(R),str(R/'experiments/ac_qualification_20261024_window01'),str(R/'experiments/ac_structure_20261024_window01'),str(R/'experiments')]
from models.optimize.ac_spatial_odd_moment_prototype_window01.spatial_odd_moment import SpatialOddNet,spatial_descriptor,ARMS
from qualification_model import QualifiedBackbone,counted_flops
import run_qualification as q
from models.optimize.ac_spatial_odd_moment_prototype_window01.c4_moment_stem import counted_flops as moment_flops
torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
save=lambda name,data:(S/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def main():
    assert os.getuid()==1000 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cfg=json.loads((S/'config.json').read_text())
    assert all(sha(R/p)==h for p,h in cfg['sources'].items())
    assert all(sha(R/p)==h for p,h in cfg['locked'].items())
    rng=np.random.default_rng(921);a=rng.normal(size=(3,5,7,7))
    x=torch.tensor(a,dtype=torch.float64,requires_grad=True)
    centered=a-a.mean((-2,-1),keepdims=True);var=(centered**2).mean((-2,-1),keepdims=True);z=centered/np.sqrt(var+1e-6)
    oracle={'odd':(np.tanh(z)**3).mean((-2,-1)), 'even':(np.tanh(z)**2).mean((-2,-1)), 'raw3':(z**3).mean((-2,-1)), 'std':np.sqrt(var[:,:,0,0]+1e-6)-.001}
    numeric={}
    for kind in oracle:
        y=spatial_descriptor(x,kind,False);expected=torch.tensor(oracle[kind])
        error=float((y-expected).abs().max().detach());assert error<1e-12
        d=oracle[kind];norm=(d-d.mean(1,keepdims=True))/np.sqrt(d.var(1,keepdims=True)+1e-5)
        assert np.max(np.abs(spatial_descriptor(x,kind).detach().numpy()-norm))<1e-11
        small=torch.randn(2,3,2,3,dtype=torch.float64,requires_grad=True)
        assert torch.autograd.gradcheck(lambda v:spatial_descriptor(v,kind),(small,),eps=1e-6,atol=3e-5,rtol=1e-3)
        flat=torch.full((2,3,1,1),4.,dtype=torch.float64,requires_grad=True)
        out=spatial_descriptor(flat,kind);out.sum().backward()
        assert torch.isfinite(out).all() and torch.isfinite(flat.grad).all() and torch.count_nonzero(out)==0
        # Statistics are independent of batch composition and spatial ordering.
        assert torch.allclose(spatial_descriptor(x[:1],kind),spatial_descriptor(x,kind)[:1],atol=1e-12,rtol=0)
        perm=torch.tensor(rng.permutation(49))
        permuted=x.flatten(-2)[:,:,perm].reshape_as(x)
        assert torch.allclose(spatial_descriptor(permuted,kind),spatial_descriptor(x,kind),atol=1e-11,rtol=0)
        numeric[kind]={'numpy_max_error':error,'double_gradcheck':True,'degenerate_finite':True,'batch_independent':True,'spatial_permutation_invariant':True}
    # Exactly equal first and second moments need not determine odd statistic.
    sym=torch.tensor([-1.,-1.,1.,1.],dtype=torch.float64).reshape(1,1,2,2)
    asym=torch.tensor([-1.,-1.,-1.,3.],dtype=torch.float64).reshape(1,1,2,2)/math.sqrt(3)
    assert abs(float(sym.mean()-asym.mean()))<1e-15 and abs(float(sym.square().mean()-asym.square().mean()))<1e-15
    sv=float(spatial_descriptor(sym,'odd',False));av=float(spatial_descriptor(asym,'odd',False))
    assert abs(sv)<1e-15 and abs(av)>1e-3
    odd=spatial_descriptor(x,'odd',False)
    assert torch.allclose(spatial_descriptor(-x,'odd',False),-odd,atol=1e-12,rtol=0)
    assert torch.allclose(spatial_descriptor(-x,'even',False),spatial_descriptor(x,'even',False),atol=1e-12,rtol=0)
    assert torch.allclose(spatial_descriptor(x+7,'odd',False),odd,atol=1e-12,rtol=0)
    scale_error=float((spatial_descriptor(2*x,'odd',False)-odd).abs().max().detach());assert scale_error<1e-6
    stress=torch.cat([torch.zeros(1,5,7,7),torch.randn(1,5,7,7)*1e6],0).requires_grad_()
    for kind in oracle:
        v=spatial_descriptor(stress,kind);g=torch.autograd.grad(v.square().sum(),stress,retain_graph=True)[0]
        assert torch.isfinite(v).all() and torch.isfinite(g).all()
    # Toy optimization checks objective-sensitive adapter updates without CLE images.
    updates={}
    for kind in oracle:
        d=spatial_descriptor(x.detach().float(),kind)
        proj=torch.nn.Linear(5,3);torch.nn.init.zeros_(proj.weight);torch.nn.init.zeros_(proj.bias)
        opt=torch.optim.SGD(proj.parameters(),lr=.01);losses=[]
        for step in range(2):
            opt.zero_grad();loss=F.cross_entropy(proj(d),torch.tensor([0,1,2]));loss.backward()
            assert torch.isfinite(loss) and proj.weight.grad.norm()>0
            losses.append(float(loss.detach()));opt.step()
        assert losses[1]<losses[0];updates[kind]=losses
    init=[];q.verify()
    for seed in [45,46,47]:
        info=q.core.read_json(q.STUDY/'pretrained_manifest.json')['assets']['convnext_tiny']
        assert sha(R/info['path'])==info['sha256']
        q.core.set_seed(seed);base=QualifiedBackbone('convnext_tiny',q.STUDY/'pretrained').eval()
        image=torch.randn(1,3,224,224)
        with torch.no_grad():expected=base(image)
        for arm in ['plain','mom_only',*ARMS]:
            old=copy.deepcopy(base);ids={id(p) for p in old.parameters()};state=torch.random.get_rng_state().clone()
            net=SpatialOddNet(old,arm).eval()
            assert torch.equal(state,torch.random.get_rng_state())
            assert ids<=set(id(p) for p in net.parameters()) and net.original_parameter_ids==ids
            with torch.no_grad():actual=net(image)
            assert torch.equal(actual,expected)
            params=sum(p.numel() for p in net.parameters());extra=(18528 if arm.startswith('mom_') else 0)+(2307 if arm in ARMS else 0)
            assert params==27822435+extra
            if seed==45 and arm in ARMS:
                net.train();net.zero_grad(set_to_none=True)
                loss=F.cross_entropy(net(image),torch.tensor([1]));loss.backward()
                grad=float(net.projection.weight.grad.norm());assert grad>0 and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            else:grad=None
            budget=moment_flops(net,image,counted_flops) if seed==45 else None
            init.append(dict(seed=seed,arm=arm,public_logit_max_error=0.,rng_preserved=True,original_parameter_objects_retained=True,parameters=params,counted_flops=budget,projection_first_gradient_norm=grad))
            del net,old
        del base
    result=dict(passed=True,numeric=numeric,same_mean_variance_counterexample={'symmetric':sv,'asymmetric':av},positive_scale_approx_error=scale_error,
                toy_two_updates=updates,public_initialization=init,source_sha256=cfg['sources'],CLE_images=0,CLE_training=0,val_images=0,test_images=0,
                scope='operator prototype only; no clinical effectiveness or innovation confirmation',overall_research_complete=False)
    save('numeric_preflight_cpu.json',result);save('complete.json',dict(passed=True,proof_sha256=sha(S/'numeric_preflight_cpu.json'),overall_research_complete=False))
    print(json.dumps({'passed':True,'public_variants':len(init),'proof_sha256':sha(S/'numeric_preflight_cpu.json')}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:save('failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
