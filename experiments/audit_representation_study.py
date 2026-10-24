"""Audit independent-pretraining probes without inference or fitting."""
from pathlib import Path
import collections
import json
import sys
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_representation_study as study
from pretrained_representations import dino,state_sha
core=study.core;cfg=study.verify();records=core.read_json(study.STUDY/'manifest.json')
jobs=core.read_json(study.STUDY/'final_jobs.json');assert len(jobs)==15
cache=core.read_json(study.CACHE/'features.json');assert cache['sha256']==core.digest(study.CACHE/'features.npz')
expected_sig=__import__('hashlib').sha256((core.digest(study.STUDY/'config.json')+'independent-pretrained-features').encode()).hexdigest()
assert cache['signature']==expected_sig
with np.load(study.CACHE/'features.npz',allow_pickle=False) as a:feature=a['features'];indices=a['indices']
assert feature.shape==(2386,1280) and feature.dtype==np.float32 and np.isfinite(feature).all()
assert np.array_equal(indices,np.arange(2386)) and cache['indices']==list(range(2386))
for name in ['resnet','dino']:
 model=study.arch.ORIGINAL_MODEL(42).cpu() if name=='resnet' else dino(ROOT)
 assert state_sha(model)==cache['encoders'][name]['state_sha256']
 del model
original=core.read_json(study.DATA/'oof_jobs.json');merged=collections.defaultdict(list);audit=[]
mapping={'resnet_frozen_gap':[np.arange(512)],'dino_cls':[np.arange(512,896)],
         'dino_mean':[np.arange(896,1280)],'dino_concat':[np.arange(512,896),np.arange(896,1280)]}
for job in jobs:
 assert study.done(job)
 old=next(j for j in original if j['id']==job['source_oof_job_id'])
 for a,b in [('fit','train'),('inner','select'),('score','score')]:assert job[a]==old[b]
 core.validate_split(records,job['fit'],job['inner'],job['score'])
 assert set(job['fit']+job['inner']+job['score'])==set(range(2386))
 out=study.STUDY/'runs'/job['id'];comp=core.read_json(out/'complete.json')
 assert comp['new_neural_trainings']==0 and comp['ridge_fits']==4
 target=np.eye(3)[[records[i]['label'] for i in job['fit']]]
 for kind,blocks in mapping.items():
  result=next(r for r in core.read_json(out/'results.json') if r['scheme']==kind)
  path=ROOT/result['head'];assert core.digest(path)==result['head_sha256']
  with np.load(path,allow_pickle=False) as z:head={k:z[k] for k in z.files}
  assert np.array_equal(head['fit_indices'],job['fit']) and all(np.isfinite(v).all() for v in head.values())
  cols=np.concatenate(blocks);assert np.array_equal(head['columns'],cols)
  raw=feature[np.asarray(job['fit'])][:,cols].astype(np.float64);mean=raw.mean(0);scale=np.maximum(raw.std(0),1e-6);start=0
  for block in blocks:scale[start:start+len(block)]*=np.sqrt(len(block)*len(blocks));start+=len(block)
  assert np.array_equal(mean,head['mean']) and np.array_equal(scale,head['scale'])
  x=(raw-mean)/scale;coef=head['coef'].T;bias=head['intercept']
  assert coef.shape==(len(cols),3) and bias.shape==(3,)
  residual=x@coef+bias-target;gradient=x.T@residual+coef
  assert np.max(np.abs(gradient))<1e-7 and np.max(np.abs(residual.sum(0)))<1e-7
  outer=(feature[np.asarray(job['score'])][:,cols].astype(np.float64)-mean)/scale
  scores=outer@coef+bias;scores-=scores.max(1,keepdims=True);p=np.exp(scores);p/=p.sum(1,keepdims=True)
  rows=study.arch.read_csv(out/(kind+'_outer_predictions.csv'));assert [int(r['index']) for r in rows]==job['score']
  saved=np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows]);assert np.allclose(p,saved,rtol=0,atol=1e-12)
  y=np.array([int(r['label']) for r in rows]);yp=saved.argmax(1)
  for r in rows:
   m=records[int(r['index'])];assert m['split']=='train' and m['sample_id']==r['sample_id'] and m['group_id']==r['group_id'] and m['label']==int(r['label'])
  assert np.array_equal(yp,[int(r['pred']) for r in rows])
  assert abs(f1_score(y,yp,labels=[0,1,2],average='macro')-result['macro_f1'])<1e-12
  assert confusion_matrix(y,yp,labels=[0,1,2]).tolist()==result['confusion_matrix']
  assert path.owner()=='login'
  merged[kind,job['seed']]+=rows
  audit.append(dict(job=job['id'],scheme=kind,head_sha256=result['head_sha256'],normal_equation_max_residual=float(np.max(np.abs(gradient)))))
 assert all(p.owner()=='login' for p in out.rglob('*') if p.is_file())
for name,encoder in [('cnn_finetuned_gap','bn_train'),('cnn_frozenbn_gap','bn_frozen')]:
 for seed in [42,43,44]:
  path=study.STUDY/'references'/f'{name}_s{seed}.csv';source=study.PREVIOUS/'oof'/f'{encoder}_gap_s{seed}.csv'
  assert core.digest(path)==core.digest(source)
  merged[name,seed]=study.arch.read_csv(path)
summary=core.read_json(study.STUDY/'round9_results.json');assert len(summary)==18 and len(audit)==60
for (scheme,seed),rows in merged.items():
 rows=sorted(rows,key=lambda r:int(r['index']));merged[scheme,seed]=rows
 assert [int(r['index']) for r in rows]==list(range(2386))
 assert study.arch.read_csv(study.STUDY/'oof'/f'{scheme}_s{seed}.csv')==rows
 result=next(r for r in summary if r['scheme']==scheme and r['seed']==seed)
 y=[int(r['label']) for r in rows];yp=[int(r['pred']) for r in rows]
 assert abs(f1_score(y,yp,labels=[0,1,2],average='macro')-result['macro_f1'])<1e-12
 assert confusion_matrix(y,yp,labels=[0,1,2]).tolist()==result['confusion_matrix']
groups=sorted({r['group_id'] for r in records if r['split']=='train'});assert len(groups)==185
weights=np.random.default_rng(20261024).multinomial(185,np.ones(185)/185,size=20000)
def matrix(rows):
 m=np.zeros((185,3,3))
 for r in rows:m[groups.index(r['group_id']),int(r['label']),int(r['pred'])]+=1
 return m
def score(m):
 d=m.sum(-1)+m.sum(-2)
 return np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),d,out=np.zeros_like(d),where=d!=0).mean(-1)
comparison=core.read_json(study.STUDY/'paired_comparisons.json')
pairs=[(s,'resnet_frozen_gap') for s in ['dino_cls','dino_mean','dino_concat']]+[('dino_concat',s) for s in study.REFS]
for candidate,baseline in pairs:
 ds=[];boot=[]
 for seed in [42,43,44]:
  a=matrix(merged[baseline,seed]);b=matrix(merged[candidate,seed]);ds.append(float(score(b.sum(0))-score(a.sum(0))))
  boot.append(score(np.einsum('bg,gij->bij',weights,b))-score(np.einsum('bg,gij->bij',weights,a)))
 row=comparison['comparisons'][candidate+'_vs_'+baseline];b=np.mean(boot,axis=0);family=np.percentile(b,[.5,99.5])
 assert np.allclose(row['seed_deltas'],ds,atol=1e-12) and np.allclose(row['patient95'],np.percentile(b,[2.5,97.5]),atol=1e-12)
 assert np.allclose(row['family_interval'],family,atol=1e-12)
 assert row['strong_pair']==(min(ds)>0 and np.mean(ds)>=.01 and family[0]>0)
assert comparison['strong_primary_followup']==all(comparison['comparisons']['dino_concat_vs_'+b]['strong_pair'] for b in ['resnet_frozen_gap']+study.REFS)
mean_metrics={s:{k:float(np.mean([r[k] for r in summary if r['scheme']==s])) for k in ['macro_f1','malignant_recall','patient_mean_accuracy']} for s in study.ALL}
core.save_json(study.STUDY/'completion_audit.json',dict(passed=True,timestamp=core.timestamp(),new_neural_trainings=0,new_ridge_fits=60,
 checks=['two pretrained weight/runtime/source hashes','global cache signature/SHA, finite original-train 2386x1280 and exact indices',
         'cached encoder state hashes reproduce source initializations','15 unchanged three-way patient splits',
         '60 fit-only standardizers and original-label convex ridge optima','60 saved-head outer predictions independently reproduced',
         'six references identical to audited prior OOF files','18 complete OOF sets and independently recomputed metrics',
         '185-patient bootstrap, five-comparison family intervals and full primary gate','login ownership'],heads=audit,mean_metrics=mean_metrics,comparison=comparison))
print(json.dumps(dict(passed=True,new_neural_trainings=0,new_ridge_fits=60,mean_metrics=mean_metrics,strong_primary_followup=comparison['strong_primary_followup']),indent=2))
