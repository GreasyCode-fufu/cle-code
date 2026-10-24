#!/usr/bin/env python3
"""Round 8: fixed-feature probes. Zero new CNN training; 30 units / 90 ridge fits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_architecture_study as arch
from texture_statistics import projection,descriptors,fit_transform
core=arch.core
DATA=ROOT/'experiments/data_study';PREVIOUS=ROOT/'experiments/bn_oof_study'
STUDY=ROOT/'experiments/statistics_study';CACHE=ROOT/'.cache_statistics_study';HEADS=ROOT/'weights/statistics_study'
KINDS=['gap','gap_std','gap_cov']
arch.STUDY=STUDY;arch.__file__=__file__;core.STUDY=STUDY
ORIGINAL_SAVE=core.save_json;ORIGINAL_COMMIT=arch.commit
def save_json(path,value):
 if Path(path)==STUDY/'status.json' and value.get('phase')=='round1_training':
  value=dict(value,phase='round8_feature_probe',round=8,new_cnn_trainings=0,total_ridge_fits=90)
 ORIGINAL_SAVE(path,value)
core.save_json=save_json
arch.commit=lambda message:ORIGINAL_COMMIT(message.replace('development-only architecture run','fixed-feature diagnostic unit'))

def sources():
 return dict(driver=core.digest(__file__),statistics=core.digest(ROOT/'experiments/texture_statistics.py'),
             protocol=core.digest(ROOT/'experiments/ROUND8_PROTOCOL.md'),trainer=core.digest(core.__file__),
             supervisor=core.digest(ROOT/'experiments/run_architecture_study.py'),backbone=core.digest(ROOT/'cle_fair_common.py'),
             previous_driver=core.digest(ROOT/'experiments/run_bn_oof_study.py'))

def verify():
 cfg=core.read_json(STUDY/'config.json')
 assert cfg['architecture_sources']==sources()
 assert core.digest(STUDY/'manifest.json')==cfg['manifest_sha256']
 assert core.digest(DATA/'config.json')==cfg['data_config_sha256']
 assert core.digest(PREVIOUS/'config.json')==cfg['previous_config_sha256']
 assert core.digest(STUDY/'final_jobs.json')==cfg['jobs_sha256']
 assert core.digest(STUDY/'projection.npy')==cfg['projection_sha256']
 assert core.digest(core.CACHE/'meta.json')==cfg['cache_meta_sha256']
 return cfg
arch.verify=verify

def source_valid(job):
 comp=core.read_json(ROOT/job['source_complete'])
 signature=hashlib.sha256((json.dumps(job['source_job'],sort_keys=True)+core.digest(ROOT/job['source_config'])+core.digest(core.__file__)).encode()).hexdigest()
 assert comp['signature']==signature==job['source_signature']
 assert core.digest(ROOT/job['source_complete'])==job['source_complete_sha256']
 assert core.digest(ROOT/job['source_predictions'])==job['source_predictions_sha256']
 assert core.digest(ROOT/job['source_checkpoint'])==job['source_checkpoint_sha256']==comp['checkpoint_sha256']
 assert comp['epochs_completed']==60
 return comp

def prepare():
 STUDY.mkdir(parents=True,exist_ok=True);CACHE.mkdir(exist_ok=True)
 if (STUDY/'config.json').exists():verify();return
 (STUDY/'manifest.json').write_bytes((DATA/'manifest.json').read_bytes())
 np.save(STUDY/'projection.npy',projection())
 original=core.read_json(DATA/'oof_jobs.json');previous=core.read_json(PREVIOUS/'final_jobs.json');jobs=[]
 for old in original:
  for arm in ['bn_train','bn_frozen']:
   src=old if arm=='bn_train' else next(j for j in previous if j['source_job_id']==old['id'])
   folder=DATA if arm=='bn_train' else PREVIOUS
   out=folder/'runs'/src['id'];comp=core.read_json(out/'complete.json')
   job=dict(id=f'r8_{arm}_s{old["seed"]}_f{old["fold"]}',arm=arm,seed=old['seed'],fold=old['fold'],
     fit=old['train'],inner=old['select'],score=old['score'],source_job=src,
     source_config=str((folder/'config.json').relative_to(ROOT)),source_complete=str((out/'complete.json').relative_to(ROOT)),
     source_complete_sha256=core.digest(out/'complete.json'),source_predictions=str((out/'oof_predictions.csv').relative_to(ROOT)),
     source_predictions_sha256=core.digest(out/'oof_predictions.csv'),source_signature=comp['signature'],
     source_checkpoint=comp['checkpoint'],source_checkpoint_sha256=comp['checkpoint_sha256'])
   source_valid(job);jobs.append(job)
 core.save_json(STUDY/'final_jobs.json',jobs)
 cfg=dict(round=8,created_at=core.timestamp(),architecture_sources=sources(),seeds=[42,43,44],kinds=KINDS,
    new_cnn_trainings=0,feature_units=30,ridge_fits=90,active_indices=list(range(2386)),
    manifest_sha256=core.digest(STUDY/'manifest.json'),data_config_sha256=core.digest(DATA/'config.json'),
    previous_config_sha256=core.digest(PREVIOUS/'config.json'),jobs_sha256=core.digest(STUDY/'final_jobs.json'),
    projection_sha256=core.digest(STUDY/'projection.npy'),cache_meta_sha256=core.digest(core.CACHE/'meta.json'),
    pretrained_sha256=core.read_json(DATA/'config.json')['pretrained_sha256'],
    descriptors=dict(gap=512,gap_std=1024,gap_cov=1040,stored_features=1552),
    ridge=dict(alpha=1.,fit_intercept=True,solver='cholesky',standardizer_fit='fit indices only',block_scaling='sqrt(block dimension times block count)'),
    inference='center224 / resize256, original train only, frozen source CNN in eval; no new CNN gradients',
    policy='No grades, fixed val/test, relabeling or hyperparameter search; ridge softmax scores are uncalibrated.',
    uncertainty='Internal development-selected cross-fit probe, not independent external validation.',max_parallel=2,cpu_threads=2)
 core.save_json(STUDY/'config.json',cfg)
 arch.commit('research: freeze six fixed-descriptor schemes and 30 reused CNN sources')

def get_features(job):
 import torch
 cfg=verify();source_valid(job);path=CACHE/(job['id']+'.npz');meta=CACHE/(job['id']+'.json');sig=core.signature(job)
 if meta.exists():
  cached=core.read_json(meta);assert cached['signature']==sig and cached['sha256']==core.digest(path)
  with np.load(path,allow_pickle=False) as arr:return arr['features'],arr['cnn_probabilities'],cached
 records=core.read_json(STUDY/'manifest.json');feature=np.zeros((2386,1552),np.float32);prob=np.zeros((2386,3),np.float32)
 model=arch.ORIGINAL_MODEL(job['seed']).eval()
 model.load_state_dict(torch.load(ROOT/job['source_checkpoint'],map_location='cpu',weights_only=True),strict=True)
 held={};handle=model.layer4.register_forward_hook(lambda module,args,value:held.update(feature=value))
 project=torch.from_numpy(np.load(STUDY/'projection.npy',allow_pickle=False)).cuda()
 started=time.time()
 # Separate loaders preserve the source outer batch sizes and image ordering.
 with torch.inference_mode():
  for ids in [job['fit'],job['score']]:
   for x,y,_,idx in core.loader(ids,False,seed=job['seed']):
    with torch.autocast('cuda',dtype=torch.float16):logits=model(x.cuda(non_blocking=True))
    values=descriptors(held['feature'],project)
    feature[idx.numpy()]=values.cpu().numpy();prob[idx.numpy()]=logits.float().softmax(1).cpu().numpy()
 handle.remove();assert np.isfinite(feature).all() and np.isfinite(prob).all()
 original=sorted(arch.read_csv(ROOT/job['source_predictions']),key=lambda r:int(r['index']))
 expected=np.array([[float(r[f'p{k}']) for k in range(3)] for r in original])
 assert [int(r['index']) for r in original]==job['score']
 diff=float(np.max(np.abs(prob[job['score']]-expected)));assert diff<2e-6,diff
 assert np.array_equal(prob[job['score']].argmax(1),[int(r['pred']) for r in original])
 temp=path.with_suffix('.tmp')
 with temp.open('wb') as f:np.savez(f,features=feature,cnn_probabilities=prob,indices=np.array(job['fit']+job['score']))
 os.replace(temp,path)
 cached=dict(signature=sig,sha256=core.digest(path),source_checkpoint_sha256=job['source_checkpoint_sha256'],
             source_probability_maxdiff=diff,feature_shape=list(feature.shape),effective_indices=job['fit']+job['score'],seconds=time.time()-started)
 core.save_json(meta,cached);del model;torch.cuda.empty_cache()
 print(f'extracted {job["id"]} rows={len(cached["effective_indices"])} seconds={cached["seconds"]:.1f}',flush=True)
 return feature,prob,cached

def worker(job_id):
 from sklearn.linear_model import Ridge
 cfg=verify();job=next(j for j in core.read_json(STUDY/'final_jobs.json') if j['id']==job_id)
 if done(job):return
 features,cnn_prob,cache=get_features(job);records=core.read_json(STUDY/'manifest.json')
 y=np.array([r['label'] for r in records[:2386]]);target=np.eye(3)[y[job['fit']]]
 out=STUDY/'runs'/job_id;out.mkdir(parents=True,exist_ok=True);weights=HEADS/job_id;weights.mkdir(parents=True,exist_ok=True)
 artifacts={};results=[]
 for kind in KINDS:
  transformed,columns,mean,scale=fit_transform(features,job['fit'],kind)
  model=Ridge(alpha=1.,fit_intercept=True,solver='cholesky').fit(transformed[job['fit']],target)
  scores=model.predict(transformed[job['score']]);scores-=scores.max(1,keepdims=True)
  p=np.exp(scores);p/=p.sum(1,keepdims=True)
  head=weights/(kind+'.npz');np.savez(head,columns=columns,mean=mean,scale=scale,coef=model.coef_,intercept=model.intercept_,fit_indices=np.array(job['fit']))
  predictions=out/(kind+'_outer_predictions.csv');core.save_csv(predictions,core.prediction_rows(p,y[job['score']],job['score']))
  result=dict(variant=job['arm']+'_'+kind,encoder=job['arm'],descriptor=kind,seed=job['seed'],fold=job['fold'],
              n=len(job['score']),fit_count=len(job['fit']),dimensions=len(columns),head=str(head.relative_to(ROOT)),head_sha256=core.digest(head),
              train_accuracy=float((model.predict(transformed[job['fit']]).argmax(1)==y[job['fit']]).mean()),**core.metrics(p,y[job['score']]))
  results.append(result);artifacts[kind]=dict(head_sha256=result['head_sha256'],head=result['head'],predictions_sha256=core.digest(predictions))
  print(f'fit {job_id}/{kind} outer_f1={result["macro_f1"]:.6f}',flush=True)
 core.save_json(out/'results.json',results)
 core.save_json(out/'complete.json',dict(signature=core.signature(job),completed_at=core.timestamp(),source_checkpoint_sha256=job['source_checkpoint_sha256'],
       cache_sha256=cache['sha256'],artifacts=artifacts,results_sha256=core.digest(out/'results.json'),new_cnn_trainings=0,ridge_fits=3))

def done(job):
 out=STUDY/'runs'/job['id'];path=out/'complete.json'
 if not path.exists():return False
 c=core.read_json(path);assert c['signature']==core.signature(job)
 assert core.digest(out/'results.json')==c['results_sha256']
 assert core.digest(CACHE/(job['id']+'.npz'))==c['cache_sha256']
 for kind,a in c['artifacts'].items():
  assert core.digest(ROOT/a['head'])==a['head_sha256']
  assert core.digest(out/(kind+'_outer_predictions.csv'))==a['predictions_sha256']
 return True
arch.done=done;arch.baseline_predictions=lambda:None

def check():
 import torch
 from sklearn.linear_model import Ridge
 cfg=verify();records=core.read_json(STUDY/'manifest.json');jobs=core.read_json(STUDY/'final_jobs.json')
 assert len(jobs)==30
 assert core.digest(core.CACHE/'resized_uint8.npy')==core.read_json(core.CACHE/'meta.json')['cache_sha256']
 for job in jobs:
  source_valid(job);core.validate_split(records,job['fit'],job['inner'],job['score'])
  assert all(0<=i<2386 for k in ['fit','inner','score'] for i in job[k])
  assert job['fit']==job['source_job']['train'] and job['inner']==job['source_job']['select'] and job['score']==job['source_job']['score']
 q=projection();assert np.allclose(q.T@q,np.eye(32),atol=2e-6) and np.array_equal(q,np.load(STUDY/'projection.npy',allow_pickle=False))
 rng=np.random.default_rng(22);a=rng.standard_normal((2,512,7,7)).astype(np.float32);flat=a.reshape(2,512,49);mu=flat.mean(-1);center=flat-mu[:,:,None]
 projected=np.einsum('cd,bcs->bds',q,center);cov=projected@projected.transpose(0,2,1)/48
 ii,jj=np.triu_indices(32);upper=cov[:,ii,jj];upper=np.sign(upper)*np.sqrt(np.abs(upper)+1e-12)*np.where(ii==jj,1.,np.sqrt(2))
 expected=np.concatenate([mu,np.sqrt((center**2).sum(-1)/48+1e-6),upper],1)
 actual=descriptors(torch.from_numpy(a),torch.from_numpy(q)).numpy();assert np.allclose(actual,expected,atol=2e-5)
 shuffled=flat[:,:,rng.permutation(49)].reshape(a.shape);other=descriptors(torch.from_numpy(shuffled),torch.from_numpy(q)).numpy()
 assert np.allclose(actual,other,atol=2e-5)
 shifted=descriptors(torch.from_numpy(a+2),torch.from_numpy(q)).numpy();assert np.allclose(shifted[:,512:],actual[:,512:],atol=2e-5)
 constant=descriptors(torch.ones(2,512,7,7),torch.from_numpy(q)).numpy();assert np.allclose(constant[:,512:1024],.001) and np.allclose(constant[:,1024:],0)
 synthetic=rng.standard_normal((24,1552));ids=list(range(16))
 for kind in KINDS:
  transformed,columns,mean,scale=fit_transform(synthetic,ids,kind)
  assert len(columns)==cfg['descriptors'][kind] and np.allclose(mean,synthetic[ids][:,columns].mean(0))
  altered=synthetic.copy();altered[16:]+=100
  _,_,mu2,scale2=fit_transform(altered,ids,kind);assert np.array_equal(mean,mu2) and np.array_equal(scale,scale2)
  target=rng.standard_normal((16,3));head=Ridge(alpha=1,fit_intercept=True,solver='cholesky').fit(transformed[ids],target)
  z=transformed[ids]-transformed[ids].mean(0);t=target-target.mean(0)
  independent=z.T@np.linalg.solve(z@z.T+np.eye(16),t)
  assert np.allclose(head.coef_.T,independent,atol=1e-8)
 for job in jobs[:2]:get_features(job)
 core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),checks=[
   '30 signed source checkpoints, paired patient splits and original-train-only indices','full original cache SHA256',
   'orthogonal label-blind projection and exact descriptor dimensions','independent NumPy moments/covariance, constant/shift/permutation invariance',
   'fit-only transform unaffected by held-out feature changes','ridge independent closed-form coefficient check',
   'both encoder strata reproduce source outer probabilities','zero CNN training or new network gradients']))
 arch.commit('test: validate fixed moments, fit-only transforms and source CNN reproduction')
 print('PASS fixed-feature probe preflight',flush=True)

def summarize():
 cfg=verify();jobs=core.read_json(STUDY/'final_jobs.json');preds={};folds=[];summary=[]
 for arm in ['bn_train','bn_frozen']:
  for kind in KINDS:
   for seed in cfg['seeds']:
    rows=[]
    for job in [j for j in jobs if j['arm']==arm and j['seed']==seed]:
     assert done(job);out=STUDY/'runs'/job['id'];folds+= [r for r in core.read_json(out/'results.json') if r['descriptor']==kind]
     rows+=arch.read_csv(out/(kind+'_outer_predictions.csv'))
    rows=sorted(rows,key=lambda r:int(r['index']));assert [int(r['index']) for r in rows]==list(range(2386))
    preds[arm,kind,seed]=rows;core.save_csv(STUDY/'oof'/f'{arm}_{kind}_s{seed}.csv',rows)
    p=np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows]);y=np.array([int(r['label']) for r in rows]);metrics=core.metrics(p,y)
    groups=sorted({r['group_id'] for r in rows});cm=np.array(metrics['confusion_matrix'])
    accuracy=float(np.mean([np.mean([r['pred']==r['label'] for r in rows if r['group_id']==g]) for g in groups]))
    summary.append(dict(encoder=arm,descriptor=kind,seed=seed,n=2386,malignant_recall=float(cm[1,1]/cm[1].sum()),patient_mean_accuracy=accuracy,**metrics))
 weights=np.random.default_rng(20261024).multinomial(len(groups),np.ones(len(groups))/len(groups),size=20000)
 def matrix(rr):
  m=np.zeros((len(groups),3,3));np.add.at(m,([groups.index(r['group_id']) for r in rr],[int(r['label']) for r in rr],[int(r['pred']) for r in rr]),1);return m
 def f1(m):
  d=m.sum(-1)+m.sum(-2);return np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),d,out=np.zeros_like(d),where=d!=0).mean(-1)
 comparisons={}
 for arm in ['bn_train','bn_frozen']:
  for kind in ['gap_std','gap_cov']:
   ds=[];bootstrap=[]
   for seed in cfg['seeds']:
    a=matrix(preds[arm,'gap',seed]);b=matrix(preds[arm,kind,seed])
    ds.append(float(f1(b.sum(0))-f1(a.sum(0))))
    bootstrap.append(f1(np.einsum('bg,gij->bij',weights,b))-f1(np.einsum('bg,gij->bij',weights,a)))
   boots=np.mean(bootstrap,axis=0);family=np.percentile(boots,[.625,99.375]).tolist()
   comparisons[arm+'_'+kind+'_vs_gap']=dict(seed_deltas=ds,mean_delta=float(np.mean(ds)),patient95=np.percentile(boots,[2.5,97.5]).tolist(),family_interval=family,
       strong_within_stratum=min(ds)>0 and np.mean(ds)>=.01 and family[0]>0)
 lines=['# Round 8 fixed-feature probes','','Zero new CNN trainings; 30 reused encoders, 90 analytic ridge fits. No grades/fixed val/test.','',
        '| Encoder | Descriptor | Mean OOF F1 |','|---|---|---:|']
 for arm in ['bn_train','bn_frozen']:
  for kind in KINDS:lines.append(f'| {arm} | {kind} | {np.mean([r["macro_f1"] for r in summary if r["encoder"]==arm and r["descriptor"]==kind]):.6f} |')
 lines+=['',json.dumps(comparisons,indent=2),'','Conditional internal diagnostic intervals; descriptor dimensions differ and scores are uncalibrated.']
 core.save_json(STUDY/'round8_results.json',summary);core.save_json(STUDY/'fold_results.json',folds);core.save_json(STUDY/'paired_comparisons.json',comparisons)
 (STUDY/'ROUND8_REPORT.md').write_text('\n'.join(lines));arch.commit('results: compare fixed first- and second-order descriptor probes')
 core.save_json(STUDY/'status.json',dict(phase='round8_complete_needs_analysis',round=8,total=30,completed=30,new_cnn_trainings=0,ridge_fits=90,updated_at=core.timestamp()))
arch.summarize=summarize

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','worker','run']);parser.add_argument('--job');args=parser.parse_args()
 if args.command=='prepare':prepare()
 elif args.command=='check':check()
 elif args.command=='worker':worker(args.job)
 else:arch.run()
