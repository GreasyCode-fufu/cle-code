"""CPU oracle: shared vs original operators, gradients, budgets and random-state preservation."""
import ast,copy,functools,json,math,os,sys,time,types
from pathlib import Path
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import run_module_structure as m
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.optimize.ac_module_window01 import haar_wavelet
from models.optimize.ac_module_window01.library_cle import LibraryNet,EMA,ECA,FaithfulSimAM,counted_flops
from models.optimize.ac_module_window01.wtconv_reference import WTConv2d
from qualification_model import QualifiedBackbone,counted_flops as base_flops

def extracted(path,names,ns):
    code=path.read_text();tree=ast.parse(code)
    for node in tree.body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names:exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns

def main():
    m.verify();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available();os.nice(10);torch.set_num_threads(2);checks=[]
    ns={'torch':torch,'nn':nn};exec((S/'upstream/official/ema.py').read_text(),ns);official=ns['EMA'](384,factor=32).double();shared=EMA(384,factor=32).double();shared.load_state_dict(official.state_dict())
    x=torch.randn(2,384,7,9,dtype=torch.double,requires_grad=True);a=official(x);b=shared(x);assert torch.equal(a,b);ga=torch.autograd.grad(a.sum(),x,retain_graph=True)[0];gb=torch.autograd.grad(b.sum(),x)[0];assert torch.equal(ga,gb);checks.append('EMA official/shared outputs and input gradients exact at factor32; shared default8 recorded')
    ns={};exec((S/'upstream/official/eca.py').read_text(),ns);official=ns['eca_layer'](768,k_size=5).double();shared=ECA(768).double();shared.conv[0].weight.data.copy_(official.conv.weight.data);x=torch.randn(2,768,7,7,dtype=torch.double,requires_grad=True)
    assert torch.equal(official(x),shared(x));assert shared.conv[0].kernel_size==(5,);checks.append('ECA adaptive k5 explicit official k5 parity; official default3 not silently equated')
    ns={};exec((S/'upstream/official/simam.py').read_text(),ns);official=ns['simam_module'](e_lambda=1e-4).double();shared=FaithfulSimAM().double()
    for x in [torch.randn(2,8,7,9,dtype=torch.double,requires_grad=True),torch.ones(2,8,7,7,dtype=torch.double,requires_grad=True)]:
        assert torch.equal(official(x),shared(x));assert torch.isfinite(torch.autograd.grad(shared(x).sum(),x)[0]).all()
    checks.append('SimAM original feature=x*shared_PSAA_gate exact; shared gate alone is not the official feature module')
    # Execute only AST function/class definitions from original WT source, no demos/imports.
    class Haar:
        def __init__(self,name):
            assert name=='db1';v=math.sqrt(.5);self.dec_lo=[v,v];self.dec_hi=[-v,v];self.rec_lo=[v,v];self.rec_hi=[v,-v]
    ns={'torch':torch,'nn':nn,'F':F,'partial':functools.partial,'pywt':types.SimpleNamespace(Wavelet=Haar)}
    extracted(S/'upstream/shared/wtconv.py',['create_wavelet_filter','wavelet_transform','inverse_wavelet_transform','WTConv2d','_ScaleModule'],ns)
    on={'wavelet':haar_wavelet};code=(S/'upstream/official/wtconv.py').read_text().replace('from .util import wavelet','');exec(code,on)
    errors=[]
    for h,w in [(14,14),(7,7),(9,11)]:
        torch.manual_seed(7);our=WTConv2d(3,3,7,wt_levels=2).double();off=on['WTConv2d'](3,3,7,wt_levels=2).double();off.load_state_dict(our.state_dict());lib=ns['WTConv2d'](3,3,7,wt_levels=2).double();lib.load_state_dict(our.state_dict())
        x=torch.randn(2,3,h,w,dtype=torch.double,requires_grad=True);a=our(x);b=off(x);c=lib(x)
        err=max(float((a-b).abs().max().detach()),float((a-c).abs().max().detach()));assert err<1e-12;errors.append(err)
        ga=torch.autograd.grad(a.square().sum(),x,retain_graph=True)[0];gb=torch.autograd.grad(b.square().sum(),x,retain_graph=True)[0];gc=torch.autograd.grad(c.square().sum(),x)[0];assert torch.allclose(ga,gb,atol=1e-12) and torch.allclose(ga,gc,atol=1e-12)
    assert all(p.requires_grad for p in our.parameters()) and set(dict(our.named_buffers()))=={'wt_filter','iwt_filter'}
    # Independent NumPy2x2 transforms, including actual float32-created coefficients.
    x=np.arange(2*3*8*10,dtype=np.float32).reshape(2,3,8,10)/19
    lo=np.array([math.sqrt(.5)]*2,dtype=np.float32);hi=np.array([math.sqrt(.5),-math.sqrt(.5)],dtype=np.float32)
    kernels=np.stack([np.outer(lo,lo),np.outer(hi,lo),np.outer(lo,hi),np.outer(hi,hi)])
    expected=np.empty((2,3,4,4,5),dtype=np.float32)
    for i in range(4):
        for j in range(5):expected[:,:,:,i,j]=np.einsum('bcuv,kuv->bck',x[:,:,2*i:2*i+2,2*j:2*j+2],kernels)
    filters,inv=haar_wavelet.create_2d_wavelet_filter('db1',3,3);actual=haar_wavelet.wavelet_2d_transform(torch.tensor(x),filters).numpy();err=float(np.max(np.abs(actual-expected)));assert err<4e-6
    rec=haar_wavelet.inverse_2d_wavelet_transform(torch.tensor(actual),inv).numpy();assert np.allclose(rec,x,atol=8e-6)
    checks.append('WTConv shared/original/runtime outputs and gradients equal on even/odd shapes; fixed filters buffers only; NumPy Haar oracle')
    # Fresh pretrained all-seed initialization uses same original parameters and head.
    budgets=[]
    for arm in ['plain',*m.ARMS]:
        m.core.set_seed(45);base=QualifiedBackbone('convnext_tiny',m.Q/'pretrained');rng=torch.get_rng_state().clone();before={id(p):p.detach().clone() for p in base.parameters()};model=LibraryNet(base,arm)
        assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(p,before[id(p)]) for p in model.parameters() if id(p) in before)
        image=torch.randn(1,3,224,224);model.eval();out=model(image);assert torch.isfinite(out).all();F.cross_entropy(out,torch.tensor([1])).backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        budgets.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),counted_flops=counted_flops(model,image,base_flops),finite_backward=True,new_construction_rng_unchanged=True))
        del base,model,before
    expected_delta=dict(plain=0,wtconv=230976,simam=0,eca=5,ema=1488,c4_moments=18528,c4_aug=0)
    for b in budgets:assert b['parameters']==27822435+expected_delta[b['arm']],b
    cfg=m.READ(S/'config.json');checks_data=[]
    for arm in ['plain','c4_aug']:
        for training in [True,False]:
            ids=cfg['indices']['train' if training else 'val'][:16];a=m.q.GuardedData(ids,training,'A');b=m.Data(ids,training,45,arm)
            for pos in range(len(ids)):
                torch.manual_seed(20261024+pos);x,y,w,i=a[pos];state=torch.get_rng_state().clone();torch.manual_seed(20261024+pos);z,yy,ww,ii=b[pos]
                assert y==yy and w==ww and i==ii and torch.equal(state,torch.get_rng_state())
                assert any(torch.equal(z,torch.rot90(x,k,(-2,-1))) for k in range(4)) if training and arm=='c4_aug' else torch.equal(x,z)
            checks_data.append(dict(arm=arm,training=training,images=len(ids),base_rgb_or_quarter_turn_exact=True,global_rng_exact=True))
    m.SAVE(S/'module_source_audit.json',dict(passed=True,device='CPU',network_training=0,test_images_loaded=0,checks=checks,wtconv_max_errors=errors,numpy_dwt_error=err,budgets=budgets,data_rng_checks=checks_data,source_hashes=m.sources(),c4_prototype_prior_cpu_checks='ac_c4_moment_prototype_window01/numeric_preflight_cpu_tiny224.json',limitation='Operator and computational parity only; not evidence of CLE model efficacy or novelty.'))
    m.commit('test: audit faithful library formulas, Haar operators, gradients, fixed budgets and RNG preservation')

if __name__=='__main__':main()
