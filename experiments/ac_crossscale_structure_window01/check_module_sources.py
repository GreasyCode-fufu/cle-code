"""Independent NumPy/official operator and initialization checks, no CLE training."""
from pathlib import Path
import ast,functools,operator,os,sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
from qualification_model import QualifiedBackbone,counted_flops as base_flops
from models.optimize.ac_crossscale_structure_window01.library_cle import LibraryNet,descriptor,counted_flops

def main():
    cfg=m.verify();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available();os.nice(10);torch.set_num_threads(2)
    gen=torch.Generator().manual_seed(16);a=torch.randn(2,4,9,generator=gen,dtype=torch.double,requires_grad=True);b=torch.randn(2,4,9,generator=gen,dtype=torch.double,requires_grad=True)
    aa=a.detach().numpy();bb=b.detach().numpy();aa=aa-aa.mean(-1,keepdims=True);bb=bb-bb.mean(-1,keepdims=True);aa/=np.sqrt((aa*aa).mean(-1,keepdims=True)+1e-4);bb/=np.sqrt((bb*bb).mean(-1,keepdims=True)+1e-4)
    z=(aa@bb.transpose(0,2,1)/9).reshape(2,-1);expected=16*z/np.sqrt(1+(z*z).sum(1,keepdims=True));np.testing.assert_allclose(descriptor(a,b).detach().numpy(),expected,rtol=1e-12,atol=1e-12)
    perm=torch.randperm(9,generator=gen);assert torch.allclose(descriptor(a,b),descriptor(a[:,:,perm],b[:,:,perm]),atol=1e-12,rtol=1e-12)
    assert not torch.allclose(descriptor(a,b),descriptor(a,b[:,:,perm]))
    assert torch.allclose(descriptor(a,b),descriptor(a+3,b-2),atol=1e-12,rtol=1e-12)
    assert torch.autograd.gradcheck(lambda x,y:descriptor(x,y),(a,b))
    constants=torch.ones(2,4,9,dtype=torch.double,requires_grad=True);descriptor(constants,constants).sum().backward();assert torch.isfinite(constants.grad).all()
    tree=ast.parse((S/'upstream/BCNN.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TensorProduct');scope=dict(torch=torch,nn=nn,functools=functools,operator=operator);exec(compile(ast.Module(body=[node],type_ignores=[]),'official_BCNN_TensorProduct','exec'),scope)
    official=scope['TensorProduct']([4,4]);x=a.reshape(2,4,3,3);y=b.reshape(2,4,3,3);z=official(x,y).flatten(1);root=(F.relu(z)+1e-5).sqrt()-(F.relu(-z)+1e-5).sqrt();reference=16*F.normalize(root,dim=1)
    assert torch.equal(reference,descriptor(a,b,'bilinear'));g1=torch.autograd.grad(reference.sum(),(a,b),retain_graph=True);g2=torch.autograd.grad(descriptor(a,b,'bilinear').sum(),(a,b));assert all(torch.equal(v,w) for v,w in zip(g1,g2))
    budgets=[];image=torch.randn(1,3,224,224);initial=None
    for kind in ['plain','moments',*m.ARMS]:
        m.core.set_seed(45);base=QualifiedBackbone('convnext_tiny',m.Q/'pretrained');state=torch.get_rng_state().clone();original={id(p):p.detach().clone() for p in base.parameters()};model=LibraryNet(base,kind).eval()
        assert torch.equal(state,torch.get_rng_state()) and all(torch.equal(p,original[id(p)]) for p in model.parameters() if id(p) in original)
        output=model(image)
        if initial is None:initial=output.detach()
        else:assert torch.equal(output,initial),kind
        F.cross_entropy(output,torch.tensor([1])).backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        expected=27822435+(0 if kind in ['plain','corr_only'] else 18528)+(19203 if kind in m.ARMS else 0);assert sum(p.numel() for p in model.parameters())==expected
        budgets.append(dict(arm=kind,parameters=expected,counted_flops=counted_flops(model,image,base_flops),initial_public_logits_exact=True,RNG_preserved=True,finite_backward=True));del base,model,original
    # Same original stochastic train transforms and val transform, exact tensors/RNG.
    for train in [True,False]:
        ids=cfg['indices']['train' if train else 'val'][:8];ref=m.q.GuardedData(ids,train,'A');data=m.Data(ids,train,45,m.ARMS[0])
        for i in range(8):
            torch.manual_seed(900+i);x,y,w,j=ref[i];state=torch.get_rng_state().clone();torch.manual_seed(900+i);xx,yy,ww,jj=data[i]
            assert torch.equal(x,xx) and (y,w,j)==(yy,ww,jj) and torch.equal(state,torch.get_rng_state())
    m.SAVE(S/'module_source_audit.json',dict(passed=True,budgets=budgets,NumPy_centered_correlation=True,bilinear_official_operator_and_gradient_exact=True,joint_permutation_and_shift_checks=True,double_gradcheck=True,constant_gradient_finite=True,CPU_initialization_and_parameter_pairing=True,train_val_transform_exact=True,network_training=0,test_images=0,scope='numerical checks only, CLE effectiveness unknown'))
    m.commit('test: verify cross-scale correlation NumPy gradients and faithful BCNN operator with paired public initialization')
if __name__=='__main__':main()
