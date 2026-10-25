from pathlib import Path
import ast,builtins,copy,hashlib,json,os,sys,traceback
import numpy as np
import torch
from torch.nn import functional as F
S=Path(__file__).resolve().parent;R=S.parents[1]
sys.path[:0]=[str(R),str(R/'experiments/ac_qualification_20261024_window01'),str(R/'experiments/ac_structure_20261024_window01'),str(R/'experiments')]
from models.optimize.ac_channel_covariance_prototype_window01.channel_covariance import covariance_descriptor,iterative_sqrt,CovarianceNet,CovarianceReadout,ARMS,counted_flops
from qualification_model import QualifiedBackbone,counted_flops as base_flops
import run_qualification as q
torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
save=lambda n,v:(S/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def ns(a,n=5):
    tr=np.trace(a,axis1=-2,axis2=-1)[:,None,None];y=a/tr;z=np.broadcast_to(np.eye(a.shape[-1]),a.shape).copy()
    for _ in range(n):t=.5*(3*np.eye(a.shape[-1])-z@y);y=y@t;z=t@z
    return y*np.sqrt(tr)

def oracle(a,kind,permutation=None):
    x=a.reshape(*a.shape[:2],-1)
    if kind=='channelperm':x=np.take_along_axis(x,permutation[None],axis=2)
    c=x-x.mean(-1,keepdims=True);cov=np.stack([sum(np.outer(row[:,i],row[:,i]) for i in range(row.shape[1]))/row.shape[1] for row in c])
    if kind=='diag':cov=cov*np.eye(c.shape[1])
    if kind!='rawcov':cov=ns(cov+1e-4*np.eye(c.shape[1]))-ns(np.broadcast_to(1e-4*np.eye(c.shape[1]),cov.shape))
    i,j=np.triu_indices(c.shape[1]);return cov[:,i,j]*np.where(i==j,1.,np.sqrt(2.))

def official():
    source=S/'references/official_MPNCOV.py';license=S/'references/LICENSE_MPNCOV'
    for p,expected in [(source,'400cdf16469d9a729fda31a4a310708fd365b895'),(license,'fcd5240305399993bb1479b8ac45cc147e77c991')]:
        b=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==expected
    assert 'MIT License' in license.read_text() and 'Peihua Li and Jiangtao Xie' in license.read_text()
    selected=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name in ['Covpool','Sqrtm']];assert len(selected)==2
    for node in selected:
        assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(node))
        assert not any(isinstance(n,ast.Name) and n.id in ['open','exec','eval','compile','__import__','os','sys','subprocess'] for n in ast.walk(node))
    env={'torch':torch,'Function':torch.autograd.Function,'__name__':'audited_official_mpn'}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),env)
    v=torch.randn(2,4,4,dtype=torch.float64);a=(v@v.transpose(1,2)+torch.eye(4)).requires_grad_();probe=torch.randn_like(a);probe=(probe+probe.transpose(1,2))/2
    y=iterative_sqrt(a);ref=env['Sqrtm'].apply(a,5)
    output_error=float((y-ref).abs().max().detach());g=torch.autograd.grad((y*probe).sum(),a)[0];gref=torch.autograd.grad((ref*probe).sum(),a)[0]
    gradient_error=float(((g+g.transpose(1,2)-gref-gref.transpose(1,2))/2).abs().max());assert output_error<1e-10 and gradient_error<1e-9
    x=torch.randn(2,4,3,3,requires_grad=True);c=x.flatten(2)-x.flatten(2).mean(-1,keepdim=True);cov=c@c.transpose(1,2)/9;cref=env['Covpool'].apply(x)
    err=float((cov-cref).abs().max().detach());p=torch.randn_like(cov);gg=torch.autograd.grad((cov*p).sum(),x)[0];gr=torch.autograd.grad((cref*p).sum(),x)[0];ge=float((gg-gr).abs().max());assert err<2e-6 and ge<2e-6
    return dict(source_sha256=sha(source),license_sha256=sha(license),restricted_AST_classes=['Covpool','Sqrtm'],sqrt_forward_error=output_error,symmetric_domain_gradient_error=gradient_error,covariance_float32_error=err,covariance_gradient_error=ge,full_paper_reproduction=False)

def main():
    assert os.getuid()==1000 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cfg=json.loads((S/'config.json').read_text());assert all(sha(R/p)==h for name in ['sources','locked'] for p,h in cfg[name].items())
    rng=np.random.default_rng(20261025);a=rng.normal(size=(2,4,7,7));x=torch.tensor(a,dtype=torch.float64,requires_grad=True);perm=np.stack([rng.permutation(49) for _ in range(4)]);tp=torch.tensor(perm,dtype=torch.long);numeric={}
    for kind in ['sqrt','rawcov','diag','channelperm']:
        d=covariance_descriptor(x,kind,False,tp);ref=oracle(a,kind,perm);err=float(np.max(np.abs(d.detach().numpy()-ref)));assert err<1e-11
        normalized=(ref-ref.mean(1,keepdims=True))/np.sqrt(ref.var(1,keepdims=True)+1e-5);assert np.max(np.abs(covariance_descriptor(x,kind,True,tp).detach().numpy()-normalized))<1e-10
        small=torch.randn(1,2,3,3,dtype=torch.float64,requires_grad=True);per2=torch.tensor(np.stack([rng.permutation(9) for _ in range(2)]),dtype=torch.long)
        assert torch.autograd.gradcheck(lambda v:covariance_descriptor(v,kind,True,per2),(small,),eps=1e-6,atol=3e-5,rtol=1e-3)
        flat=torch.ones(2,4,7,7,dtype=torch.float64,requires_grad=True);out=covariance_descriptor(flat,kind,permutation=tp);out.square().sum().backward();assert out.abs().max()<1e-10 and torch.isfinite(flat.grad).all()
        assert torch.allclose(covariance_descriptor(x[:1],kind,permutation=tp),covariance_descriptor(x,kind,permutation=tp)[:1],atol=1e-12,rtol=0)
        numeric[kind]=dict(numpy_error=err,double_gradcheck=True,constant_zero_finite=True,batch_independent=True)
    for h,w in [(1,1),(1,7),(7,1),(7,9),(3,3)]:
        a=torch.randn(2,4,h,w,dtype=torch.float64,requires_grad=True)
        for kind in ['sqrt','rawcov','diag']:
            d=covariance_descriptor(a,kind);g=torch.autograd.grad(d.square().sum(),a,retain_graph=True)[0];assert torch.isfinite(d).all() and torch.isfinite(g).all()
            assert torch.allclose(d,covariance_descriptor(torch.rot90(a,1,(-2,-1)),kind),atol=1e-10,rtol=0)
    shift=torch.randn(1,4,1,1,dtype=torch.float64)
    assert torch.allclose(covariance_descriptor(x+shift),covariance_descriptor(x),atol=1e-11,rtol=0)
    layout=torch.tensor([[[-1.,-1.,1.,1.],[-1.,-1.,1.,1.]]],dtype=torch.float64).reshape(1,2,2,2);flipped=layout.clone();flipped[:,1]*=-1
    assert torch.equal(layout.flatten(2).sort().values,flipped.flatten(2).sort().values)
    d1=covariance_descriptor(layout,'sqrt',False);d2=covariance_descriptor(flipped,'sqrt',False);difference=float((d1-d2).abs().max());assert difference>.1
    assert torch.equal(covariance_descriptor(layout,'diag'),covariance_descriptor(flipped,'diag'))
    c=(x.detach().numpy().reshape(2,4,49));c=c-c.mean(-1,keepdims=True);c=c@c.transpose(0,2,1)/49+1e-4*np.eye(4);ev,evec=np.linalg.eigh(c);root=(evec*np.sqrt(ev)[:,None,:])@evec.transpose(0,2,1)
    errors={str(n):float(np.max(np.abs(ns(c,n)-root))) for n in [5,10,20]};assert errors['10']<=errors['5']+1e-12 and errors['20']<1e-10
    off=official();toy={}
    for kind in ['sqrt','rawcov','diag','channelperm']:
        net=CovarianceReadout(kind);v=torch.randn(2,768,7,7);opt=torch.optim.SGD(net.parameters(),lr=.01);records=[]
        for step in range(2):
            opt.zero_grad();loss=F.cross_entropy(net(v),torch.tensor([0,1]));loss.backward();hn=float(net.classifier.weight.grad.norm());pn=float(net.project.weight.grad.norm());assert hn>0 and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            assert pn==0 if step==0 else pn>0
            records.append(dict(loss=float(loss.detach()),classifier_grad=hn,project_grad=pn));opt.step()
        assert records[1]['loss']<records[0]['loss'];toy[kind]=records
    init=[];q.verify()
    for seed in [45,46,47]:
        q.core.set_seed(seed);base=QualifiedBackbone('convnext_tiny',q.STUDY/'pretrained').eval();image=torch.randn(1,3,224,224)
        with torch.no_grad():expected=base(image)
        for arm in ['plain','mom_only',*ARMS]:
            old=copy.deepcopy(base);ids={id(p) for p in old.parameters()};state=torch.random.get_rng_state().clone();net=CovarianceNet(old,arm).eval();assert torch.equal(state,torch.random.get_rng_state()) and ids<=set(id(p) for p in net.parameters())
            with torch.no_grad():actual=net(image)
            assert torch.equal(actual,expected);count=sum(p.numel() for p in net.parameters());target=27822435+(18528 if arm.startswith('mom_') else 0)+(12699 if arm in ARMS else 0);assert count==target
            grad=None
            if seed==45 and arm in ARMS:
                net.train();net.zero_grad();F.cross_entropy(net(image),torch.tensor([1])).backward();grad=float(net.readout.classifier.weight.grad.norm());assert grad>0 and float(net.readout.project.weight.grad.norm())==0 and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            budget=counted_flops(net,image,base_flops) if seed==45 else None
            if budget is not None:
                extra=1204224+816+25088+(0 if arm=='mom_rawcov' else 122880) if arm in ARMS else 0
                assert budget==8909531136+(202309632 if arm.startswith('mom_') else 0)+extra,(arm,budget,extra)
            init.append(dict(seed=seed,arm=arm,parameters=count,counted_flops=budget,initial_logits_exact=True,RNG_preserved=True,original_parameter_objects_retained=True,classifier_first_grad=grad));del net,old
        del base
    result=dict(passed=True,numeric=numeric,official_reference=off,equal_marginals_distinct_joint_difference=difference,eig_oracle_iteration_errors=errors,finite5_not_exact_square_root=True,toy_two_updates=toy,public_initialization=init,CLE_images=0,CLE_training=0,val_images=0,test_images=0,source_sha256=cfg['sources'],innovation_confirmed=False,overall_research_complete=False)
    save('numeric_preflight_cpu.json',result);save('complete.json',dict(passed=True,proof_sha256=sha(S/'numeric_preflight_cpu.json'),overall_research_complete=False));print(json.dumps(dict(passed=True,public_variants=len(init),proof_sha256=sha(S/'numeric_preflight_cpu.json'))),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:save('failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
