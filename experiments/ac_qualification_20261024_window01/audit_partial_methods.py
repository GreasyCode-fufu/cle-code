"""CPU equation audit: FasterNet channel PConv versus NVIDIA masked PartialConv."""
from pathlib import Path
import ast,hashlib,json,urllib.request,datetime
import torch
from torch import nn
import numpy as np

S=Path('/home/login/data/cle/auto/GPT/experiments/ac_qualification_20261024_window01')
D=S/'references_official';D.mkdir(exist_ok=True)
sources={
 'fasternet.py':'https://raw.githubusercontent.com/JierunChen/FasterNet/master/models/fasternet.py',
 'partialconv2d.py':'https://raw.githubusercontent.com/NVIDIA/partialconv/master/models/partialconv2d.py',
 'FasterNet_LICENSE':'https://raw.githubusercontent.com/JierunChen/FasterNet/master/LICENSE',
 'PartialConv_LICENSE':'https://raw.githubusercontent.com/NVIDIA/partialconv/master/LICENSE.txt',
}
records={}
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:12789'}))
for name,url in sources.items():
 p=D/name
 try:
  if not p.exists():p.write_bytes(opener.open(url,timeout=15).read())
  records[name]=dict(url=url,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
 except Exception as e:
  records[name]=dict(url=url,fetch_error=repr(e));assert 'LICENSE' in name,'required method source download failed'

def one_class(path,name):
 text=path.read_text(encoding='utf8');node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name==name)
 namespace=dict(torch=torch,nn=nn,Tensor=torch.Tensor)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
 return namespace[name]

shared=next(Path('/home/login/data/cle/auto/shared/module_library').glob('21.*Partial Convolution.py'))
Shared=one_class(shared,'Partial_conv3');Official=one_class(D/'fasternet.py','Partial_conv3')
torch.manual_seed(20261024);checks=[]
for channels,division in [(16,4),(17,4),(32,2)]:
 a=Shared(channels,division,'split_cat').double();b=Official(channels,division,'split_cat').double()
 b.load_state_dict(a.state_dict(),strict=True)
 x=torch.randn(2,channels,9,11,dtype=torch.float64,requires_grad=True);y=x.detach().clone().requires_grad_(True)
 out=a(x);reference=b(y);assert torch.equal(out,reference)
 out.square().sum().backward();reference.square().sum().backward()
 assert torch.equal(x.grad,y.grad) and torch.equal(a.partial_conv3.weight.grad,b.partial_conv3.weight.grad)
 assert torch.equal(out[:,a.dim_conv3:],x[:,a.dim_conv3:])
 checks.append(dict(channels=channels,division=division,forward_exact=True,input_gradient_exact=True,weight_gradient_exact=True,untouched_channels_exact=True))

# Independent NumPy check of bias-corrected support normalization, including an empty field.
input=np.arange(1,26,dtype=np.float64).reshape(5,5);mask=np.ones((5,5));mask[:2,:2]=0
kernel=np.array([[1.,2.,1.],[0.,1.,0.],[-1.,0.,1.]])*.1;bias=.3
x=torch.tensor(input)[None,None];m=torch.tensor(mask)[None,None];w=torch.tensor(kernel)[None,None]
raw=nn.functional.conv2d(x*m,w,bias=torch.tensor([bias],dtype=torch.float64),padding=1)
count=nn.functional.conv2d(m,torch.ones_like(w),padding=1)
ratio=9/(count+1e-8);valid=count.clamp(0,1)
actual=((raw-bias)*ratio*valid+bias)*valid
xp=np.pad(input*mask,1);mp=np.pad(mask,1);expected=np.zeros_like(input)
for i in range(5):
 for j in range(5):
  total=mp[i:i+3,j:j+3].sum()
  if total:expected[i,j]=(xp[i:i+3,j:j+3]*kernel).sum()*9/(total+1e-8)+bias
assert np.allclose(actual[0,0].numpy(),expected,atol=1e-12)
zero=nn.functional.conv2d(torch.zeros_like(m),torch.ones_like(w),padding=1)
assert torch.equal(zero.clamp(0,1),torch.zeros_like(zero))
report=dict(passed=True,timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 source_fetch_note='Initial direct raw.githubusercontent.com read timed out; existing SSH-configured proxy succeeded, no proxy/other process modified.',
 source_files=records,shared_path=str(shared),shared_sha256=hashlib.sha256(shared.read_bytes()).hexdigest(),
 faster_channel_pconv_checks=checks,masked_partialconv_bias_and_boundary_numpy_max_error=float(np.max(np.abs(actual[0,0].numpy()-expected))),
 masked_partialconv_empty_support_valid_zero=True,
 meaning='Shared21 implements FasterNet channel-partial convolution, not support-normalized masked PartialConv. Only its class AST executed in an isolated namespace; no import/cache write in shared library.',
 limitation='NVIDIA operator formula checked against independent scalar reference; no claim of whole inpainting network reproduction or CLE performance.',
 future='Any optical-field method must preserve the fixed AC images/labels and distinguish geometric support from darkness; architecture not selected until qualification baseline freeze.')
(S/'partial_method_equation_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(report,ensure_ascii=False))
