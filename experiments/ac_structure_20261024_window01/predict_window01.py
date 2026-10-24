"""Research-only inference from the explicitly named arm and paired seed."""
import argparse
from pathlib import Path
import hashlib
import json
import sys
import torch
from PIL import Image
from torchvision import transforms as T

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
from models.optimize.cle_ac_window01 import CLEDepthNet,ConvNeXtReference

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['baseline','relational','no_neighbor','no_depth','final_only','official4','convnext'])
p.add_argument('--seed',type=int,default=45,choices=[45,46,47]);p.add_argument('--image',type=Path,required=True)
p.add_argument('--device',default='cuda',choices=['cuda','cpu']);p.add_argument('--output',type=Path)
a=p.parse_args()
cfg=json.loads((HERE/'config.json').read_text())
for name,s in cfg['source_hashes'].items():assert sha(ROOT/name)==s,'frozen source changed: '+name
folder=HERE/'references'/f's{a.seed}' if a.arm=='baseline' else HERE/'runs'/f'window01_{a.arm}_s{a.seed}'
comp=json.loads((folder/'complete.json').read_text())
checkpoint=ROOT/comp['checkpoint'];assert sha(checkpoint)==comp['checkpoint_sha256']
torch.set_num_threads(1)
model=ConvNeXtReference(ROOT) if a.arm=='convnext' else CLEDepthNet(ROOT,a.arm)
model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
model.to(a.device).eval()
image=Image.open(a.image).convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
transform=T.Compose([T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
x=transform(image)[None].to(a.device)
with torch.inference_mode():
    if a.device=='cuda':
        with torch.autocast('cuda',dtype=torch.float16):logits=model(x)
    else:logits=model(x)
    prob=logits.float().softmax(-1)[0].cpu().tolist()
names=['Benign_lesions','Malignant_lesions','body_of_stomach_normal']
result=dict(arm=a.arm,seed=a.seed,image=str(a.image),checkpoint_sha256=comp['checkpoint_sha256'],
    class_order=names,probabilities=prob,prediction=names[max(range(3),key=prob.__getitem__)],
    note='研究模型；未经过本新任务最终test确认或临床概率校准。CPU为FP32，CUDA为FP16。')
if a.output:
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
