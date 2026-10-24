"""Independent fixed-feature audit; no new image inference or model fitting."""
from pathlib import Path
import collections
import json
import sys
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_statistics_study as study
core=study.core;cfg=study.verify();records=core.read_json(study.STUDY/'manifest.json')
jobs=core.read_json(study.STUDY/'final_jobs.json');assert len(jobs)==30
merged=collections.defaultdict(list);audits=[];source_diff=[]
for job in jobs:
 comp=study.source_valid(job);assert study.done(job)
 assert all(0<=i<2386 for key in ['fit','inner','score'] for i in job[key])
 assert set(job['fit']+job['inner']+job['score'])==set(range(2386))
 core.validate_split(records,job['fit'],job['inner'],job['score'])
 for key,source_key in [('fit','train'),('inner','select'),('score','score')]:assert job[key]==job['source_job'][source_key]
 cached=core.read_json(study.CACHE/(job['id']+'.json'))
 assert cached['signature']==core.signature(job) and cached['effective_indices']==job['fit']+job['score']
 path=study.CACHE/(job['id']+'.npz');assert core.digest(path)==cached['sha256']
 with np.load(path,allow_pickle=False) as arr:
  feature=arr['features'];cnn=arr['cnn_probabilities'];indices=arr['indices']
 assert feature.shape==(2386,1552) and cnn.shape==(2386,3) and np.isfinite(feature).all() and np.isfinite(cnn).all()
 assert np.array_equal(indices,job['fit']+job['score'])
 assert not feature[job['inner']].any() and not cnn[job['inner']].any()
 source=study.arch.read_csv(ROOT/job['source_predictions'])
 assert [int(r['index']) for r in source]==job['score']
 p0=np.array([[float(r[f'p{k}']) for k in range(3)] for r in source])
 diff=float(np.max(np.abs(cnn[job['score']]-p0)));assert diff<2e-6 and abs(diff-cached['source_probability_maxdiff'])<1e-12
 assert np.array_equal(cnn[job['score']].argmax(1),[int(r['pred']) for r in source])
 source_diff.append(diff)
 out=study.STUDY/'runs'/job['id'];unit=core.read_json(out/'complete.json')
 assert unit['new_cnn_trainings']==0 and unit['ridge_fits']==3
 fit_y=np.array([records[i]['label'] for i in job['fit']]);target=np.eye(3)[fit_y]
 for kind in study.KINDS:
  result=next(r for r in core.read_json(out/'results.json') if r['descriptor']==kind)
  path=ROOT/result['head'];assert core.digest(path)==result['head_sha256']
  with np.load(path,allow_pickle=False) as h:head={k:h[k] for k in h.files}
  assert set(head)=={'columns','mean','scale','coef','intercept','fit_indices'}
  assert np.array_equal(head['fit_indices'],job['fit']) and all(np.isfinite(v).all() for v in head.values())
  columns=np.arange(512) if kind=='gap' else np.r_[np.arange(512),np.arange(512,1024) if kind=='gap_std' else np.arange(1024,1552)]
  assert np.array_equal(head['columns'],columns)
  raw=feature[np.asarray(job['fit'])][:,columns].astype(np.float64)
  mean=raw.mean(0);scale=np.maximum(raw.std(0),1e-6)
  dims=[512] if kind=='gap' else [512,len(columns)-512]
  start=0
  for width in dims:scale[start:start+width]*=np.sqrt(width*len(dims));start+=width
  assert np.array_equal(head['mean'],mean) and np.array_equal(head['scale'],scale)
  x=(raw-mean)/scale;coef=head['coef'].T;intercept=head['intercept']
  assert coef.shape==(len(columns),3) and intercept.shape==(3,)
  residual=x@coef+intercept-target
  # Unique ridge optimum verified without repeating any completed fit.
  assert np.max(np.abs(x.T@residual+coef))<1e-7 and np.max(np.abs(residual.sum(0)))<1e-7
  xo=(feature[np.asarray(job['score'])][:,columns].astype(np.float64)-mean)/scale
  z=xo@coef+intercept;z-=z.max(1,keepdims=True);prob=np.exp(z);prob/=prob.sum(1,keepdims=True)
  rows=study.arch.read_csv(out/(kind+'_outer_predictions.csv'))
  assert [int(r['index']) for r in rows]==job['score']
  stored=np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows])
  assert np.allclose(prob,stored,rtol=0,atol=1e-12) and np.allclose(stored.sum(1),1,atol=1e-12)
  y=np.array([int(r['label']) for r in rows]);yp=stored.argmax(1)
  for r in rows:
   m=records[int(r['index'])];assert m['split']=='train' and r['sample_id']==m['sample_id'] and r['group_id']==m['group_id'] and int(r['label'])==m['label']
  assert np.array_equal(yp,[int(r['pred']) for r in rows])
  assert abs(f1_score(y,yp,labels=[0,1,2],average='macro',zero_division=0)-result['macro_f1'])<1e-12
  assert confusion_matrix(y,yp,labels=[0,1,2]).tolist()==result['confusion_matrix']
  assert path.owner()=='login'
  merged[job['arm'],kind,job['seed']]+=rows
  audits.append(dict(unit=job['id'],descriptor=kind,fit_count=len(job['fit']),outer_count=len(rows),head_sha256=result['head_sha256'],ridge_normal_equation_max_residual=float(np.max(np.abs(x.T@residual+coef)))))
 assert all(q.owner()=='login' for q in out.rglob('*') if q.is_file())
 assert (study.CACHE/(job['id']+'.npz')).owner()=='login'
summary=core.read_json(study.STUDY/'round8_results.json');assert len(summary)==18 and len(audits)==90
for (arm,kind,seed),rows in merged.items():
 rows=sorted(rows,key=lambda r:int(r['index']));merged[arm,kind,seed]=rows
 assert [int(r['index']) for r in rows]==list(range(2386))
 assert study.arch.read_csv(study.STUDY/'oof'/f'{arm}_{kind}_s{seed}.csv')==rows
 r=next(r for r in summary if r['encoder']==arm and r['descriptor']==kind and r['seed']==seed)
 assert abs(f1_score([int(v['label']) for v in rows],[int(v['pred']) for v in rows],labels=[0,1,2],average='macro')-r['macro_f1'])<1e-12
groups=sorted({r['group_id'] for r in records if r['split']=='train'});assert len(groups)==185
weights=np.random.default_rng(20261024).multinomial(185,np.ones(185)/185,size=20000)
def matrix(rows):
 m=np.zeros((185,3,3))
 for r in rows:m[groups.index(r['group_id']),int(r['label']),int(r['pred'])]+=1
 return m
def score(m):
 denominator=m.sum(-1)+m.sum(-2)
 return np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),denominator,out=np.zeros_like(denominator),where=denominator!=0).mean(-1)
comparisons=core.read_json(study.STUDY/'paired_comparisons.json')
for arm in ['bn_train','bn_frozen']:
 for kind in ['gap_std','gap_cov']:
  deltas=[];boots=[]
  for seed in [42,43,44]:
   a=matrix(merged[arm,'gap',seed]);b=matrix(merged[arm,kind,seed]);deltas.append(float(score(b.sum(0))-score(a.sum(0))))
   boots.append(score(np.einsum('bg,gij->bij',weights,b))-score(np.einsum('bg,gij->bij',weights,a)))
  boot=np.mean(boots,axis=0);r=comparisons[arm+'_'+kind+'_vs_gap']
  assert np.allclose(r['seed_deltas'],deltas,atol=1e-12) and np.allclose(r['patient95'],np.percentile(boot,[2.5,97.5]),atol=1e-12)
  family=np.percentile(boot,[.625,99.375]);assert np.allclose(r['family_interval'],family,atol=1e-12)
  assert r['strong_within_stratum']==(min(deltas)>0 and np.mean(deltas)>=.01 and family[0]>0)
core.save_json(study.STUDY/'completion_audit.json',dict(passed=True,timestamp=core.timestamp(),reused_cnn_sources=30,new_cnn_trainings=0,ridge_fits=90,
 checks=['30 source checkpoint signatures and SHA','feature-cache signature/SHA/effective original-train indices','source outer CNN probabilities reproduced',
         '90 fit-only transform moments and exact fit indices','90 convex ridge normal equations without refitting','saved head prediction/metrics reproduced',
         'patient isolation and unchanged source folds','18 x 2386 merged OOF rows, three predictions per image/scheme',
         '185-patient bootstrap and four-comparison family intervals independently recomputed','login ownership'],
 max_source_probability_diff=max(source_diff),heads=audits,comparisons=comparisons))
print(json.dumps(dict(passed=True,new_cnn_trainings=0,ridge_fits=90,max_source_probability_diff=max(source_diff),comparisons=comparisons),indent=2))
