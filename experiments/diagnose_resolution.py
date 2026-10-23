"""Frozen-checkpoint scale sensitivity, development images only."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_data_study as core
from models.optimize.cle_scale_route import ScaleRouteResNet
OUT=ROOT/'experiments/scale_study/resolution_diagnostic'
OUT.mkdir(parents=True,exist_ok=True)
cfg=core.read_json(core.STUDY/'config.json')
records=core.read_json(core.STUDY/'manifest.json')
indices=cfg['val_indices']
assert len(indices)==538 and {records[i]['split'] for i in indices}=={'val'}
protocol=dict(source_sha256=core.digest(__file__),seeds=[42,43,44],models=['gap','dilated_control'],
              evaluation_scales=[dict(crop=336,resize=384),dict(crop=448,resize=512)],
              data_manifest_sha256=cfg['manifest_sha256'],selection='No training and no test. Original 224 results reused.',
              interpretation='Input scale changes distribution and effective receptive fields. This does not estimate performance after high-resolution training.')
core.save_json(OUT/'protocol.json',protocol)


class Images(torch.utils.data.Dataset):
 def __init__(self,crop,resize):
  self.resize=resize
  self.transform=T.Compose([T.CenterCrop(crop),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 def __len__(self):return len(indices)
 def __getitem__(self,i):
  idx=indices[i];r=records[idx]
  with Image.open(ROOT/r['path']) as im:
   x=im.convert('RGB').resize((self.resize,self.resize),Image.Resampling.BILINEAR)
  return self.transform(x),r['label'],1.,idx


def rows(path):
 with open(path) as f:return list(csv.DictReader(f))


predictions={};results=[]
for kind in protocol['models']:
 for seed in protocol['seeds']:
  core.set_seed(seed)
  folder=ROOT/'experiments'/('data_study' if kind=='gap' else 'scale_study')
  job=f'grade_weighted_s{seed}' if kind=='gap' else f'r3_grade_weighted_dilated_control_s{seed}'
  completion=core.read_json(folder/'runs'/job/'complete.json')
  ck=ROOT/completion['checkpoint'];assert core.digest(ck)==completion['checkpoint_sha256']
  if kind=='gap':old=ROOT/'experiments/scale_study/baselines'/f'grade_weighted_s{seed}'/'validation_predictions.csv'
  else:old=folder/'runs'/job/'validation_predictions.csv'
  predictions[kind,seed,224]=rows(old)
  results.append(dict(model=kind,seed=seed,crop=224,source='reused frozen predictions',
    macro_f1=core.metrics(np.array([[float(r[f'p{k}']) for k in range(3)] for r in predictions[kind,seed,224]]),
                         np.array([int(r['label']) for r in predictions[kind,seed,224]]))['macro_f1']))
  model=core.new_model(seed)
  if kind!='gap':model=ScaleRouteResNet(model,'dilated_control').cuda()
  model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=True)
  for scale in protocol['evaluation_scales']:
   crop,resize=scale['crop'],scale['resize']
   target=OUT/f'{kind}_s{seed}_{crop}'
   sig=hashlib.sha256((json.dumps(protocol,sort_keys=True)+completion['checkpoint_sha256']+str(crop)).encode()).hexdigest()
   if (target/'complete.json').exists() and core.read_json(target/'complete.json')['signature']==sig:
    r=core.read_json(target/'complete.json')
   else:
    dl=torch.utils.data.DataLoader(Images(crop,resize),batch_size=24,num_workers=2,shuffle=False,pin_memory=True)
    p,y,idx=core.infer(model,dl)
    core.save_csv(target/'predictions.csv',core.prediction_rows(p,y,idx))
    r=dict(model=kind,seed=seed,crop=crop,resize=resize,n=len(y),signature=sig,
           checkpoint_sha256=completion['checkpoint_sha256'],**core.metrics(p,y))
    core.save_json(target/'complete.json',r)
   predictions[kind,seed,crop]=rows(target/'predictions.csv');results.append(r)
   print(json.dumps(dict(model=kind,seed=seed,crop=crop,f1=r['macro_f1'])),flush=True)
  del model;torch.cuda.empty_cache()

groups=sorted({records[i]['group_id'] for i in indices})
bw=np.random.default_rng(20261024).multinomial(len(groups),np.ones(len(groups))/len(groups),size=20000)
def cm(rr):
 rr=sorted(rr,key=lambda r:int(r['index']))
 assert [int(r['index']) for r in rr]==sorted(indices)
 m=np.zeros((len(groups),3,3))
 np.add.at(m,([groups.index(r['group_id']) for r in rr],[int(r['label']) for r in rr],[int(r['pred']) for r in rr]),1)
 return m
def f1(m):
 d=m.sum(-1)+m.sum(-2)
 return np.mean(np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),d,out=np.zeros_like(d),where=d!=0),axis=-1)
summary=[]
for kind in protocol['models']:
 for crop in [336,448]:
  ds=[];boot=[]
  for seed in protocol['seeds']:
   ma,mb=cm(predictions[kind,seed,224]),cm(predictions[kind,seed,crop])
   ds.append(float(f1(mb.sum(0))-f1(ma.sum(0))))
   boot.append(f1(np.einsum('bg,gij->bij',bw,mb))-f1(np.einsum('bg,gij->bij',bw,ma)))
  summary.append(dict(model=kind,crop=crop,mean_f1=float(np.mean([r['macro_f1'] for r in results if r['model']==kind and r['crop']==crop])),
                      seed_deltas=ds,mean_delta=float(np.mean(ds)),patient_cluster95=np.percentile(np.mean(boot,axis=0),[2.5,97.5]).tolist()))
core.save_json(OUT/'results.json',results);core.save_json(OUT/'summary.json',summary)
print(json.dumps(summary,indent=2),flush=True)
