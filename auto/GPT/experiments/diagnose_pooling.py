"""Development-only spatial pooling diagnostic. Never uses test images."""
from pathlib import Path
import sys,json
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import run_data_study as core
DATA=ROOT/'experiments/data_study'
OUT=ROOT/'experiments/architecture_study/pooling_diagnostic.json'

def main():
 cfg=core.read_json(DATA/'config.json');ids=cfg['val_indices'];summary=[]
 for recipe in ['ac_only','grade_weighted']:
  for seed in [42,43,44]:
   meta=core.read_json(DATA/'runs'/f'{recipe}_s{seed}'/'complete.json')
   model=core.new_model(seed).eval()
   ck=ROOT/meta['checkpoint'];assert core.digest(ck)==meta['checkpoint_sha256']
   model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=True)
   scores={k:[] for k in ['gap','gem2','gem3','class_top20']};ys=[];support=[]
   with torch.inference_mode():
    for x,y,_,idx in core.loader(ids,False,seed=seed):
     x=x.cuda()
     with torch.autocast('cuda',dtype=torch.float16):
      z=model.maxpool(model.relu(model.bn1(model.conv1(x))))
      z=model.layer4(model.layer3(model.layer2(model.layer1(z))))
      gap=model.fc(z.mean((2,3)))
      g2=model.fc(z.clamp_min(1e-6).square().mean((2,3)).sqrt())
      g3=model.fc(z.clamp_min(1e-6).pow(3).mean((2,3)).pow(1/3))
      spatial=F.conv2d(z,model.fc.weight[:,:,None,None],model.fc.bias).flatten(2)
      top=spatial.topk(max(1,round(spatial.shape[-1]*.2)),dim=2).values.mean(2)
     for k,v in [('gap',gap),('gem2',g2),('gem3',g3),('class_top20',top)]:
      scores[k].append(v.float().softmax(1).cpu().numpy())
     pred=gap.argmax(1).cpu();patch=spatial.argmax(1).cpu()
     support.extend([dict(index=int(i),label=int(t),gap_correct=bool(p==t),
                          spatial_correct_fraction=float((pp==t).float().mean()))
                     for i,t,p,pp in zip(idx,y,pred,patch)])
     ys.extend(y.tolist())
   labels=np.array(ys)
   record=dict(recipe=recipe,seed=seed,checkpoint_sha256=meta['checkpoint_sha256'],n=len(labels),
               metrics={k:core.metrics(np.concatenate(v),labels) for k,v in scores.items()},
               erroneous_images=sum(not r['gap_correct'] for r in support),
               erroneous_with_some_correct_locations=sum(not r['gap_correct'] and r['spatial_correct_fraction']>0 for r in support),
               support=support)
   summary.append(record)
   print(recipe,seed,{k:round(v['macro_f1'],4) for k,v in record['metrics'].items()},flush=True)
   del model;torch.cuda.empty_cache()
 core.save_json(OUT,dict(records=summary,created_at=core.timestamp(),
    interpretation='Post-hoc responses of fixed models only; feature locations have overlapping receptive fields, not independent histological annotations. Does not prove retrained pooling efficacy.'))

if __name__=='__main__':main()
