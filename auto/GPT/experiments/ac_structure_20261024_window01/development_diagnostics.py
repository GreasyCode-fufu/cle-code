"""CPU-only train/val evidence; no new model fitting or test access."""
from pathlib import Path
import collections,csv,json,hashlib,datetime
import numpy as np
from PIL import Image

R=Path('/home/login/data/cle/auto/GPT');S=R/'experiments/ac_structure_20261024_window01'
cfg=json.loads((S/'config.json').read_text());records=json.loads((S/'manifest.json').read_text())
array=np.load(R/'.cache_data_study/resized_uint8.npy',mmap_mode='r')
assert array.shape==(3595,256,256,3)
predictions={}
for arm in ['baseline','relational','no_neighbor','no_depth','final_only','official4','convnext']:
 rows=[]
 for seed in [45,46,47]:
  folder=S/'references'/f's{seed}' if arm=='baseline' else S/'runs'/f'window01_{arm}_s{seed}'
  p=folder/'validation_predictions.csv'
  if not p.exists():break
  x=list(csv.DictReader(p.open()));assert [int(r['index']) for r in x]==cfg['indices']['val'];rows.append(x)
 if len(rows)==3:predictions[arm]=rows
features=[]
for split in ['train','val']:
 for i in cfg['indices'][split]:
  assert records[i]['split']==split
  raw=array[i];gray=raw.astype(np.float32).mean(-1)/255
  dy=gray[1:]-gray[:-1];dx=gray[:,1:]-gray[:,:-1]
  features.append(dict(index=i,split=split,label=records[i]['label'],group_id=records[i]['group_id'],
   rgb_channel_max_difference=int(np.abs(raw[:,:,0].astype(np.int16)-raw[:,:,2].astype(np.int16)).max()),
   mean=float(gray.mean()),std=float(gray.std()),black_fraction=float((gray<.02).mean()),
   clipped_high_fraction=float((gray>.98).mean()),gradient_energy=float((dx*dx).mean()+(dy*dy).mean())))
out=S/'development_diagnostics';out.mkdir(exist_ok=True)
with (out/'allowed_image_features.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(features[0]));w.writeheader();w.writerows(features)
groups=collections.Counter(records[i]['group_id'] for i in cfg['indices']['train'])
report=dict(timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 no_test_images_accessed=True,train_groups=dict(min_images=min(groups.values()),max_images=max(groups.values()),
 top10_total=sum(v for _,v in groups.most_common(10)),total=sum(groups.values())),arms={})
errors=[]
for arm,rows in predictions.items():
 byclass={};bypatient=collections.defaultdict(list)
 counts=[]
 for pos,i in enumerate(cfg['indices']['val']):
  count=sum(int(rows[k][pos]['label'])!=int(rows[k][pos]['pred']) for k in range(3));counts.append(count)
  bypatient[records[i]['group_id']].append(count)
  errors.append(dict(arm=arm,index=i,label=records[i]['label'],group_id=records[i]['group_id'],
   seed_error_count=count,filename=Path(records[i]['path']).name))
 for label in [0,1,2]:
  idx=[p for p,i in enumerate(cfg['indices']['val']) if records[i]['label']==label]
  byclass[str(label)]=dict(n=len(idx),seed_total_errors=sum(counts[p] for p in idx),
   mean_error_fraction=sum(counts[p] for p in idx)/(3*len(idx)),all_seed_errors=sum(counts[p]==3 for p in idx))
 report['arms'][arm]=dict(class_errors=byclass,total_errors_across_seeds=sum(counts),
   error_count_histogram=dict(collections.Counter(counts)),patients_with_any_error=sum(any(v) for v in bypatient.values()),
   highest_patient_error_counts=sorted([(g,sum(v),len(v)) for g,v in bypatient.items()],key=lambda x:x[1],reverse=True)[:10])
if errors:
 with (out/'validation_error_evidence.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(errors[0]));w.writeheader();w.writerows(errors)
report['image_summary']={}
for split in ['train','val']:
 rows=[x for x in features if x['split']==split]
 report['image_summary'][split]={k:np.percentile([v[k] for v in rows],[0,25,50,75,100]).tolist()
  for k in ['rgb_channel_max_difference','mean','std','black_fraction','clipped_high_fraction','gradient_energy']}
report['limits']='Descriptive train/val evidence under frozen AC selection. Filename subtype not truth, no relabeling/selection change. Feature-error association not causal.'
(out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(report,ensure_ascii=False)[:10000])
