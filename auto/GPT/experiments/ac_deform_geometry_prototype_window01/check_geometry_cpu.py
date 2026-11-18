from pathlib import Path
import ast,copy,math,os,sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
S=Path(__file__).resolve().parent;sys.path[:0]=[str(S),str(S.parents[1]),str(S.parents[1]/'experiments/ac_qualification_20261024_window01')]
from affine_geometry import AffineResidualDepthwise,affine_offsets,deform_response

def oracle_grid(x,off,w):
    b,c,h,ww=x.shape;k=w.shape[-1];n=k*k;pad=k//2;grid=[]
    yy,xx=torch.meshgrid(torch.arange(h,dtype=x.dtype),torch.arange(ww,dtype=x.dtype),indexing='ij')
    for i in range(n):
        y=yy+i//k-pad+off[:,2*i];z=xx+i%k-pad+off[:,2*i+1];grid.append(torch.stack((2*z/(ww-1)-1,2*y/(h-1)-1),-1))
    grid=torch.stack(grid,1).reshape(b,n*h,ww,2);sample=F.grid_sample(x,grid,mode='bilinear',padding_mode='zeros',align_corners=True).reshape(b,c,n,h,ww)
    return (sample*w[:,0].flatten(1)[None,:,:,None,None]).sum(2)
def oracle_numpy(x,offset,weight):
    b,c,h,w=x.shape;k=weight.shape[-1];out=np.zeros_like(x)
    for batch in range(b):
     for ch in range(c):
      for y in range(h):
       for z in range(w):
        for ky in range(k):
         for kx in range(k):
          slot=ky*k+kx;yy=y+ky-k//2+offset[batch,2*slot,y,z];xx=z+kx-k//2+offset[batch,2*slot+1,y,z];iy=math.floor(yy);ix=math.floor(xx)
          for dy in [0,1]:
           for dx in [0,1]:
            a=iy+dy;bb=ix+dx
            if 0<=a<h and 0<=bb<w:out[batch,ch,y,z]+=x[batch,ch,a,bb]*(1-abs(yy-a))*(1-abs(xx-bb))*weight[ch,0,ky,kx]
    return out
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available();os.nice(10);torch.set_num_threads(2);torch.manual_seed(45)
    x=torch.randn(1,2,4,5,dtype=torch.double,requires_grad=True);weight=torch.randn(2,1,3,3,dtype=torch.double,requires_grad=True);off=(torch.randn(1,18,4,5,dtype=torch.double)*.15+.13).requires_grad_();actual=deform_response(x,off,weight,1);grid=oracle_grid(x,off,weight)
    np.testing.assert_allclose(actual.detach().numpy(),oracle_numpy(x.detach().numpy(),off.detach().numpy(),weight.detach().numpy()),rtol=1e-12,atol=1e-12);torch.testing.assert_close(actual,grid,rtol=1e-12,atol=1e-12)
    ga=torch.autograd.grad(actual.square().sum(),(x,off,weight),retain_graph=True);gg=torch.autograd.grad(grid.square().sum(),(x,off,weight));assert all(torch.allclose(a,b,atol=1e-10,rtol=1e-10) for a,b in zip(ga,gg))
    raw=(torch.randn(1,6,3,4,dtype=torch.double)*.1+.07).requires_grad_();xx=torch.randn(1,2,3,4,dtype=torch.double);ww=torch.randn(2,1,3,3,dtype=torch.double)
    assert torch.autograd.gradcheck(lambda p:deform_response(xx,affine_offsets(p,3),ww,1),(raw,),atol=1e-4,rtol=1e-3)
    random_raw=torch.randn(4,6,3,4,dtype=torch.double)*4;offset,j=affine_offsets(random_raw,7,return_jacobian=True);det=j[:,0,0]*j[:,1,1]-j[:,0,1]*j[:,1,0];assert float(det.min())>0 and float(det.min())>=1/2.25-1e-12 and float(det.max())<=2.25+1e-12
    operator=[]
    for kind in ['affine','permuted_affine','free']:
        base=nn.Conv2d(2,2,3,padding=1,groups=2).double();state=torch.get_rng_state().clone();model=AffineResidualDepthwise(base,kind).double();assert torch.equal(state,torch.get_rng_state()) and model.base is base
        x=torch.randn(1,2,4,5,dtype=torch.double,requires_grad=True);expected=base(x);out=model(x);assert torch.equal(out,expected)
        out.square().mean().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()) and float(model.predictor.weight.grad.norm())>0
        if kind=='permuted_affine':
            o=model.offsets(random_raw);a=affine_offsets(random_raw,3).reshape(4,9,2,3,4).index_select(1,model.spatial_permutation).flatten(1,2);assert torch.equal(a,o)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
        for _ in range(2):opt.zero_grad();loss=model(x).square().mean();loss.backward();assert torch.isfinite(loss);opt.step()
        operator.append(dict(kind=kind,parameters=sum(p.numel() for p in model.parameters()),zero_output_exact=True,RNG_preserved=True,synthetic_two_updates_finite=True))
    # Compare author/shared source classes without executing demos/global warning filters.
    def load(path,name):
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name);scope=dict(nn=nn,torch=torch,math=math,rearrange=rearrange);exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope);return scope[name]
    official=load(S/'upstream/official_ldconv.py','LDConv');shared=load(S/'upstream/shared_ldconv.py','LDConv_2D');torch.manual_seed(45);a=official(2,3,5).double().eval();torch.manual_seed(45);b=shared(2,3,5).double().eval();aa=torch.randn(1,2,4,5,dtype=torch.double,requires_grad=True);bb=aa.detach().clone().requires_grad_();ya=a(aa);yb=b(bb);assert torch.allclose(ya,yb,atol=1e-12,rtol=1e-12);ya.sum().backward();yb.sum().backward();assert torch.allclose(aa.grad,bb.grad,atol=1e-12,rtol=1e-12)
    without=copy.deepcopy(a);without.p_conv._backward_hooks.clear();a.zero_grad();without.zero_grad();p=aa.detach().clone().requires_grad_();q=p.detach().clone().requires_grad_();a(p).sum().backward();without(q).sum().backward();assert torch.equal(a.p_conv.weight.grad,without.p_conv.weight.grad)
    corner=official(1,1,1).double().eval();nn.init.zeros_(corner.p_conv.bias);captured=[];hook=corner.conv.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()));corner(torch.ones(1,1,3,3,dtype=torch.double));hook.remove();assert captured[0][0,0,-1,-1]==4 and captured[0][0,0,0,0]==1
    # Real public pretrained network, synthetic RGB only; no CLE images.
    from qualification_model import QualifiedBackbone
    from models.optimize.ac_module_window01.c4_moment_stem import C4MomentStem
    torch.manual_seed(45);net=QualifiedBackbone('convnext_tiny',S.parents[1]/'experiments/ac_qualification_20261024_window01/pretrained').eval();net.encoder[0][0][0]=C4MomentStem(net.encoder[0][0][0],'moments');image=torch.randn(1,3,224,224)
    with torch.no_grad():before=net(image)
    base=net.encoder[0][5][-1].block[0];ids={id(p) for p in net.parameters()};state=torch.get_rng_state().clone();net.encoder[0][5][-1].block[0]=AffineResidualDepthwise(base,'affine');assert torch.equal(state,torch.get_rng_state()) and ids<={id(p) for p in net.parameters()}
    after=net(image);assert torch.equal(before,after);after.square().mean().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters());assert sum(p.numel() for p in net.parameters())==27861705
    result=dict(passed=True,official_deform_NumPy_output=True,independent_grid_sample_output_and_three_gradients=True,affine_double_gradcheck=True,local_determinant_range=[float(det.min()),float(det.max())],local_orientation_preserving_not_global_topology=True,operators=operator,shared_LDConv_vs_official_output_input_gradient=True,LDConv_actual_predictor_bias_nonzero=bool(torch.count_nonzero(a.p_conv.bias)),LDConv_hook_parameter_gradient_unchanged=True,LDConv_zero_offset_corner_weight_sum=4,LDConv_corner_interior_weight_sum=1,public_Mom_Tiny224_zero_logits_exact=True,public_parameter_ids_and_RNG_preserved=True,public_finite_backward=True,public_parameters=27861705,neural_CLE_training=0,CLE_images=0,test_predictions=0,overall_research_complete=False)
    (S/'numeric_preflight_cpu.json').write_text(__import__('json').dumps(result,indent=2)+'\n');print(__import__('json').dumps(result),flush=True)
if __name__=='__main__':main()
