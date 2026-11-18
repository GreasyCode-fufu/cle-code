"""CPU-only operator/reference/augmentation preflight; no CLE model fitting."""
from pathlib import Path
import ast,copy,hashlib,inspect,json,sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.models.swin_transformer import ShiftedWindowAttention,shifted_window_attention
import torchvision
S=Path(__file__).resolve().parent;R=S.parents[1]
sys.path.insert(0,str(S))
from field_ops_v2 import unit_circle,partial_conv,weighted_pool,window_attention,ResidualFieldConv,ResidualFieldAttention,_partition
torch.set_num_threads(1);torch.manual_seed(20261024)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def numpy_partial(base,x,mask,cap):
 z=x.detach().numpy();m=mask.numpy();w=base.weight.detach().numpy();bias=base.bias.detach().numpy() if base.bias is not None else np.zeros(base.out_channels)
 b,cin,h,wi=z.shape;kh,kw=base.kernel_size;sy,sx=base.stride;py,px=base.padding;dy,dx=base.dilation
 oh=(h+2*py-dy*(kh-1)-1)//sy+1;ow=(wi+2*px-dx*(kw-1)-1)//sx+1
 out=np.zeros((b,base.out_channels,oh,ow));cpg=cin//base.groups;opg=base.out_channels//base.groups
 for bi in range(b):
  for oc in range(base.out_channels):
   start=(oc//opg)*cpg
   for iy in range(oh):
    for ix in range(ow):
     count=0.;value=0.
     for ky in range(kh):
      for kx in range(kw):
       yy=iy*sy-py+ky*dy;xx=ix*sx-px+kx*dx
       if 0<=yy<h and 0<=xx<wi:
        support=m[bi,0,yy,xx];count+=support
        value+=sum(z[bi,start+ic,yy,xx]*support*w[oc,ic,ky,kx] for ic in range(cpg))
     valid=min(count,1.)
     ratio=kh*kw/(count+1e-8)*valid if count>0 else 0.
     if cap is not None:ratio=min(ratio,cap)
     out[bi,oc,iy,ix]=(value*ratio+bias[oc])*valid if base.bias is not None else value*ratio
 return out

def official_partial_class():
 source=R/'experiments/ac_qualification_20261024_window01/references_official/partialconv2d.py'
 node=next(n for n in ast.parse(source.read_text()) .body if isinstance(n,ast.ClassDef) and n.name=='PartialConv2d')
 ns=dict(torch=torch,nn=nn,F=F);exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
 return ns['PartialConv2d'],source

checks=[];Official,official_path=official_partial_class()
for groups,stride,dilation,bias in [(1,1,1,True),(2,2,1,True),(4,1,2,False)]:
 base=nn.Conv2d(4,4,3,stride=stride,padding=dilation,dilation=dilation,groups=groups,bias=bias).double()
 for mode in ['circle','zero','one','fractional']:
  x=torch.randn(2,4,7,9,dtype=torch.float64,requires_grad=True)
  mask=unit_circle(7,9,dtype=torch.float64).expand(2,-1,-1,-1).clone()
  if mode=='zero':mask.zero_()
  elif mode=='one':mask.fill_(1)
  elif mode=='fractional':mask.mul_(.03)
  for cap in [None,4.]:
   actual=partial_conv(base,x,mask,cap=cap)
   expected=numpy_partial(base,x,mask,cap);error=float(np.abs(actual.detach().numpy()-expected).max())
   assert error<1e-10,(groups,stride,dilation,bias,mode,cap,error)
   actual.square().mean().backward();assert torch.isfinite(x.grad).all()
   if mode=='zero':assert torch.equal(actual,torch.zeros_like(actual))
  ref=Official(4,4,3,stride=stride,padding=dilation,dilation=dilation,groups=groups,bias=bias).double()
  ref.load_state_dict(base.state_dict(),strict=True)
  a=partial_conv(base,x,mask,cap=None);b=ref(x,mask)
  e=float((a-b).abs().max().detach());assert e<1e-10
  checks.append(dict(operator='partial_conv',groups=groups,stride=stride,dilation=dilation,bias=bias,mask=mode,numpy_max_error=error,official_max_error=e))
base=nn.Conv2d(2,2,3,padding=1,groups=2).double();x=torch.randn(1,2,4,5,dtype=torch.float64,requires_grad=True);mask=unit_circle(4,5,dtype=torch.float64)
assert torch.autograd.gradcheck(lambda z:partial_conv(base,z,mask,cap=4.),(x,),fast_mode=True)
checks.append(dict(operator='partial_conv',input_gradcheck=True))
adapter=ResidualFieldConv(copy.deepcopy(base)).double()
assert torch.equal(adapter(x,mask),adapter.base(x))
opt=torch.optim.SGD(adapter.parameters(),lr=.01)
for step in range(2):
 opt.zero_grad();loss=adapter(x,mask).square().mean();loss.backward()
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.parameters())
 if step==0:assert float(adapter.alpha.grad.abs())>0
 opt.step()
assert float(adapter.alpha.detach().abs())>0
checks.append(dict(operator='partial_conv',zero_residual_exact=True,alpha_gradient_nonzero=True,finite_two_toy_updates=True))

for h,w,shift in [(14,14,[0,0]),(14,14,[3,3]),(13,17,[3,3]),(5,6,[3,3])]:
 attn=ShiftedWindowAttention(12,[7,7],shift,3).double().eval()
 x=torch.randn(2,h,w,12,dtype=torch.float64,requires_grad=True);z=x.detach().clone().requires_grad_(True)
 original=attn(x);ours=window_attention(attn,z)
 error=float((ours-original).abs().max().detach());assert error<1e-10
 ga=torch.autograd.grad(original.square().sum(),x)[0];gb=torch.autograd.grad(ours.square().sum(),z)[0]
 graderror=float((ga-gb).abs().max());assert graderror<1e-9
 support=unit_circle(h,w,dtype=torch.float64).expand(2,-1,-1,-1).clone();support[1].zero_()
 adapted=window_attention(attn,x,support)
 assert torch.isfinite(adapted).all() and torch.equal(adapted[1],torch.zeros_like(adapted[1]))
 adapted.square().sum().backward();assert torch.isfinite(x.grad).all()
 # Independent reference: add support log weights to torchvision's public relative-bias input.
 ph=(-h)%7;pw=(-w)%7;hp=h+ph;wp=w+pw;sh,sw=shift
 if hp<=7:sh=0
 if wp<=7:sw=0
 m=F.pad(support.permute(0,2,3,1),(0,0,0,pw,0,ph))
 if sh or sw:m=torch.roll(m,(-sh,-sw),(1,2))
 weights=_partition(m,7,7)[...,0];nonempty=weights.sum(-1,keepdim=True)>0
 logw=torch.where(weights>0,weights.clamp_min(1e-300).log(),torch.full_like(weights,-torch.inf));logw=torch.where(nonempty,logw,torch.zeros_like(logw))
 bias=attn.get_relative_position_bias()+logw[:,None,None,:]
 oracle=shifted_window_attention(x,attn.qkv.weight,attn.proj.weight,bias,[7,7],3,shift,qkv_bias=attn.qkv.bias,proj_bias=attn.proj.bias,training=False)
 oracle=oracle*(support.permute(0,2,3,1)>0)
 oracle_error=float((adapted-oracle).abs().max().detach());assert oracle_error<1e-10
 wrapped=ResidualFieldAttention(attn).double()
 assert torch.equal(wrapped(x,support),attn(x))
 wrapped(x,support).square().mean().backward();assert torch.isfinite(wrapped.alpha.grad) and float(wrapped.alpha.grad.abs())>0
 checks.append(dict(operator='window_attention',height=h,width=w,shift=shift,original_forward_error=error,
  original_input_gradient_error=graderror,official_weighted_bias_oracle_error=oracle_error,empty_window_zero=True,zero_residual_exact=True,alpha_gradient_nonzero=True))

x=torch.randn(2,4,9,11,dtype=torch.float64);mask=unit_circle(9,11,dtype=torch.float64).expand(2,-1,-1,-1).clone();mask[1].zero_()
pooled=weighted_pool(x,mask);expected=np.zeros((2,4,1,1))
for bi in range(2):
 if mask[bi].sum():expected[bi,:,0,0]=(x[bi].numpy()*mask[bi].numpy()).sum((1,2))/mask[bi].numpy().sum()
assert np.max(np.abs(pooled.numpy()-expected))<1e-12
checks.append(dict(operator='weighted_pool',numpy_exact=True,empty_support_zero=True))

# RGB must be unchanged by carrying an alpha support map through the same augmentation draws.
cfg=json.loads((R/'experiments/ac_dataset_v1/dataset_config.json').read_text());allowed=cfg['indices']['train'][:4]
records=json.loads((R/'experiments/ac_dataset_v1/dataset_manifest.json').read_text());byindex={r['index']:r for r in records}
augtests=0;maxdiff=0.
for resize,crop in [(256,224),(448,384)]:
 transform=T.Compose([T.RandomHorizontalFlip(),T.RandomVerticalFlip(),T.RandomRotation(15),T.RandomCrop(crop),T.ToTensor()])
 m=(unit_circle(resize,resize)[0,0].numpy()*255).astype(np.uint8)
 for index in allowed:
  row=byindex[index];assert row['split']=='train' and row['grade'] in ['A','B','C']
  with Image.open(R/row['path']) as original:rgb=original.convert('RGB').resize((resize,resize),Image.Resampling.BILINEAR)
  rgba=rgb.copy();rgba.putalpha(Image.fromarray(m))
  for seed in range(20):
   torch.manual_seed(seed);a=transform(rgb)
   torch.manual_seed(seed);b=transform(rgba)
   d=float((a-b[:3]).abs().max());maxdiff=max(maxdiff,d);assert torch.equal(a,b[:3]),(resize,index,seed,d)
   assert b.shape==(4,crop,crop) and float(b[3].min())>=0 and float(b[3].max())<=1
   augtests+=1
checks.append(dict(operator='paired_image_and_mask_augmentation',cases=augtests,raw_rgb_max_error=maxdiff,all_exact=True,
 allowed_ac_train_indices=allowed,test_images_accessed=0))
report=dict(passed=True,scope='CPU numeric prototypes only; zero CLE classifier training or candidate selection',
 seed=20261024,checks=checks,torch=torch.__version__,torchvision=torchvision.__version__,
 source_sha256={p.name:sha(p) for p in [S/'field_ops_v2.py',S/'check_field_ops_v2.py',S/'FIELD_PROTOTYPE_PROTOCOL_V2_ZH.md']},
 nvidia_source_sha256=sha(official_path),torchvision_swin_source_sha256=sha(inspect.getsourcefile(ShiftedWindowAttention)),
 limitation='Actual selected-backbone integration and real batch32 CUDA/AMP/full baseline val equivalence remain required before formal structure training.')
(S/'numeric_preflight_v2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(passed=True,checks=len(checks),paired_augmentations=augtests,test_images=0,cle_trainings=0),ensure_ascii=False))
