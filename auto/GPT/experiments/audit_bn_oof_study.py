"""Audit paired OOF artifacts without retraining or new image inference."""
from pathlib import Path
import collections
import json
import sys
import numpy as np
import torch
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_bn_oof_study as study
core=study.core;cfg=study.verify();records=core.read_json(study.STUDY/'manifest.json')
jobs=core.read_json(study.STUDY/'final_jobs.json');assert len(jobs)==15
initial_model=study.ORIGINAL_MODEL(42).cpu();initial=initial_model.state_dict()
bn_keys=[k for k in initial if 'running_' in k or 'num_batches_tracked' in k]
assert len(bn_keys)==108
audit=[];merged=collections.defaultdict(list)
for job in jobs:
 old=next(v['source_job'] for v in core.read_json(study.STUDY/'baseline_index.json') if v['source_job']['id']==job['source_job_id'])
 for name in ['train','select','score','seed','fold','epochs']:assert job[name]==old[name]
 assert job['phase']=='oof' and not job.get('sample_weights')
 assert all(0<=i<2386 for name in ['train','select','score'] for i in job[name])
 core.validate_split(records,job['train'],job['select'],job['score'])
 assert {records[i]['split'] for name in ['train','select','score'] for i in job[name]}=={'train'}
 assert study.done(job)
 for arm,key in [('bn_train',old['id']),('bn_frozen',job['id'])]:
  out=study.STUDY/('baselines' if arm=='bn_train' else 'runs')/key
  comp=core.read_json(out/'complete.json')
  expected=study.source_signature(old) if arm=='bn_train' else core.signature(job)
  assert comp['signature']==expected
  checkpoint=ROOT/comp['checkpoint'];assert core.digest(checkpoint)==comp['checkpoint_sha256']
  state=torch.load(checkpoint,map_location='cpu',weights_only=True)
  assert set(state)==set(initial) and all(torch.isfinite(v).all() for v in state.values())
  if arm=='bn_frozen':assert all(torch.equal(state[k],initial[k]) for k in bn_keys)
  log=study.arch.read_csv(out/'train_log.csv')
  assert [int(r['epoch']) for r in log]==list(range(60)) and comp['epochs_completed']==60
  assert all(np.isfinite(float(v)) for r in log for v in r.values())
  best=max(log,key=lambda r:float(r['val_macro_f1']))
  assert int(best['epoch'])==comp['best_epoch'] and abs(float(best['val_macro_f1'])-comp['best_select_f1'])<1e-12
  rows,metrics=study.validate_predictions(out/'oof_predictions.csv',job['score'],records)
  y=np.array([int(r['label']) for r in rows]);yp=np.array([int(r['pred']) for r in rows])
  independent_f1=f1_score(y,yp,labels=[0,1,2],average='macro',zero_division=0)
  assert abs(independent_f1-comp['outer_metrics']['macro_f1'])<1e-12
  assert confusion_matrix(y,yp,labels=[0,1,2]).tolist()==comp['outer_metrics']['confusion_matrix']
  if arm=='bn_frozen':
   outer=core.read_json(out/'outer_complete.json')
   assert outer['checkpoint_sha256']==comp['checkpoint_sha256']
   assert outer['prediction_sha256']==core.digest(out/'oof_predictions.csv')
  assert checkpoint.owner()=='login' and all(p.owner()=='login' for p in out.rglob('*') if p.is_file())
  merged[arm,job['seed']]+=rows
  audit.append(dict(variant=arm,seed=job['seed'],fold=job['fold'],fit=len(job['train']),inner=len(job['select']),outer=len(rows),
                    checkpoint_sha256=comp['checkpoint_sha256'],best_inner_epoch=comp['best_epoch'],outer_f1=independent_f1))
predictions={};summaries=[]
for (arm,seed),rows in merged.items():
 rows=sorted(rows,key=lambda r:int(r['index']))
 assert [int(r['index']) for r in rows]==list(range(2386))
 assert study.arch.read_csv(study.STUDY/'oof'/f'{arm}_s{seed}.csv')==rows
 predictions[arm,seed]=rows
 y=[int(r['label']) for r in rows];yp=[int(r['pred']) for r in rows]
 cm=confusion_matrix(y,yp,labels=[0,1,2])
 f1=f1_score(y,yp,labels=[0,1,2],average='macro',zero_division=0)
 result=next(r for r in core.read_json(study.STUDY/'round7_results.json') if r['variant']==arm and r['seed']==seed)
 assert abs(f1-result['macro_f1'])<1e-12 and cm.tolist()==result['confusion_matrix']
 summaries.append(dict(variant=arm,seed=seed,macro_f1=f1,malignant_recall=float(cm[1,1]/cm[1].sum())))
groups=sorted({r['group_id'] for r in records if r['split']=='train'});assert len(groups)==185
weights=np.random.default_rng(20261024).multinomial(len(groups),np.ones(len(groups))/len(groups),size=20000)
def matrix(rows):
 m=np.zeros((185,3,3))
 for r in rows:m[groups.index(r['group_id']),int(r['label']),int(r['pred'])]+=1
 return m
def score(m):
 denominator=m.sum(-1)+m.sum(-2)
 return np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),denominator,out=np.zeros_like(denominator),where=denominator!=0).mean(-1)
deltas=[];bootstrap=[]
for seed in [42,43,44]:
 a=matrix(predictions['bn_train',seed]);b=matrix(predictions['bn_frozen',seed])
 deltas.append(float(score(b.sum(0))-score(a.sum(0))))
 bootstrap.append(score(np.einsum('bg,gij->bij',weights,b))-score(np.einsum('bg,gij->bij',weights,a)))
interval=np.percentile(np.mean(bootstrap,axis=0),[2.5,97.5])
comparison=core.read_json(study.STUDY/'paired_comparison.json')
assert np.allclose(comparison['seed_deltas'],deltas,atol=1e-12) and np.allclose(comparison['patient_cluster95'],interval,atol=1e-12)
assert comparison['strong_diagnostic_signal']==(min(deltas)>0 and np.mean(deltas)>=.01 and interval[0]>0)
core.save_json(study.STUDY/'completion_audit.json',dict(passed=True,timestamp=core.timestamp(),new_models=15,reused_models=15,
 checks=['immutable source and copied baseline hashes','30 checkpoints with signed provenance and finite states','30 x 60 epochs and correct best inner selection',
         'paired original folds unchanged, only train indices, three-way patient isolation','frozen BN buffers exactly equal pretrained initialization',
         'outer indices, probabilities, independent F1/confusion and prediction SHA','2386 OOF rows per algorithm/seed, three predictions per image/algorithm',
         'independently recomputed 185-patient paired bootstrap','login ownership'],jobs=audit,comparison=comparison,metrics=summaries))
print(json.dumps(dict(passed=True,new_models=15,reused_models=15,comparison=comparison,metrics=summaries),indent=2))
