#!/usr/bin/env python3
"""Round9: independent pretrained encoders, 60 ridge fits, no neural training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_architecture_study as arch
from pretrained_representations import dino,DINO_SHA,runtime,state_sha,transform,SCHEMES
core=arch.core;DATA=ROOT/'experiments/data_study';PREVIOUS=ROOT/'experiments/statistics_study'
STUDY=ROOT/'experiments/representation_study';CACHE=ROOT/'.cache_representation_study';HEADS=ROOT/'weights/representation_study'
NEW=list(SCHEMES);REFS=['cnn_finetuned_gap','cnn_frozenbn_gap'];ALL=NEW+REFS
arch.STUDY=STUDY;arch.__file__=__file__;core.STUDY=STUDY
SAVE=core.save_json;COMMIT=arch.commit
def save_json(path,value):
 if Path(path)==STUDY/'status.json' and value.get('phase')=='round1_training':
  value=dict(value,phase='round9_representation_probe',round=9,new_neural_trainings=0,total_ridge_fits=60)
 SAVE(path,value)
core.save_json=save_json
arch.commit=lambda message:COMMIT(message.replace('development-only architecture run','pretrained-representation fold unit'))

def sources():
 return dict(driver=core.digest(__file__),encoders=core.digest(ROOT/'experiments/pretrained_representations.py'),
     protocol=core.digest(ROOT/'experiments/ROUND9_PROTOCOL.md'),license=core.digest(ROOT/'experiments/DINOV2_LICENSE.txt'),
     trainer=core.digest(core.__file__),supervisor=core.digest(ROOT/'experiments/run_architecture_study.py'),
     backbone=core.digest(ROOT/'cle_fair_common.py'),runtime=runtime())

def verify():
 cfg=core.read_json(STUDY/'config.json');assert cfg['architecture_sources']==sources()
 for path,key in [(STUDY/'manifest.json','manifest_sha256'),(DATA/'config.json','data_config_sha256'),
                  (DATA/'oof_jobs.json','source_folds_sha256'),(PREVIOUS/'config.json','previous_config_sha256'),
                  (STUDY/'final_jobs.json','jobs_sha256'),(core.CACHE/'meta.json','cache_meta_sha256')]:assert core.digest(path)==cfg[key]
 assert core.digest(ROOT/'weights/pretrained/dinov2_vits14_pretrain.pth')==cfg['dinov2_sha256']==DINO_SHA
 assert core.digest(core.PRETRAINED)==cfg['resnet_pretrained_sha256']
 for path,sha in cfg['reference_hashes'].items():assert core.digest(STUDY/path)==sha
 return cfg
arch.verify=verify

def prepare():
 STUDY.mkdir(parents=True,exist_ok=True);CACHE.mkdir(exist_ok=True)
 if (STUDY/'config.json').exists():verify();return
 (STUDY/'manifest.json').write_bytes((DATA/'manifest.json').read_bytes())
 jobs=[]
 for old in core.read_json(DATA/'oof_jobs.json'):
  jobs.append(dict(id=f'r9_s{old["seed"]}_f{old["fold"]}',seed=old['seed'],fold=old['fold'],
                   fit=old['train'],inner=old['select'],score=old['score'],source_oof_job_id=old['id']))
 core.save_json(STUDY/'final_jobs.json',jobs)
 hashes={}
 for name,source_arm in [('cnn_finetuned_gap','bn_train'),('cnn_frozenbn_gap','bn_frozen')]:
  for seed in [42,43,44]:
   source=PREVIOUS/'oof'/f'{source_arm}_gap_s{seed}.csv';target=STUDY/'references'/f'{name}_s{seed}.csv'
   target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target);hashes[str(target.relative_to(STUDY))]=core.digest(target)
 cfg=dict(round=9,created_at=core.timestamp(),architecture_sources=sources(),seeds=[42,43,44],schemes=ALL,new_schemes=NEW,
   new_neural_trainings=0,new_ridge_fits=60,fold_units=15,active_indices=list(range(2386)),
   manifest_sha256=core.digest(STUDY/'manifest.json'),data_config_sha256=core.digest(DATA/'config.json'),
   source_folds_sha256=core.digest(DATA/'oof_jobs.json'),previous_config_sha256=core.digest(PREVIOUS/'config.json'),
   jobs_sha256=core.digest(STUDY/'final_jobs.json'),cache_meta_sha256=core.digest(core.CACHE/'meta.json'),
   dinov2_sha256=DINO_SHA,resnet_pretrained_sha256=core.read_json(DATA/'config.json')['pretrained_sha256'],reference_hashes=hashes,
   source=dict(dinov2_weights='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
               license_blob='5471dc10377b76db85a2feca7a99a7eef4980ba8',model_card_blob='21b9bf295c8cab14e782e1a7a1d051be9e501088',backbones_blob='af5fd95d38eb33f73a85b92a94a715773a4e70bc'),
   extraction='FP32, TF32 disabled, center224/resize256, ImageNet normalization, eval and requires_grad=False',
   heads='Fit-only moments; per-block dimension scaling; Ridge(alpha1,intercept,cholesky), no tuning or grades.',
   interpretation='Fixed pretrained features shared across fold seeds; seeds index patient partitions, not backbone training seeds.',
   policy='No original fixed val/test access or relabeling; not new architecture, pretraining/architecture/precision/compute confounded.')
 core.save_json(STUDY/'config.json',cfg);arch.commit('research: freeze independent-pretraining probes and six copied finetuned OOF references')

def extract():
 import torch
 cfg=verify();path=CACHE/'features.npz';meta=CACHE/'features.json'
 signature=hashlib.sha256((core.digest(STUDY/'config.json')+'independent-pretrained-features').encode()).hexdigest()
 if meta.exists():
  c=core.read_json(meta);assert c['signature']==signature and core.digest(path)==c['sha256']
  with np.load(path,allow_pickle=False) as a:return a['features'],c
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 feature=np.zeros((2386,1280),np.float32);details={};started=time.time()
 for kind in ['resnet','dino']:
  model=(arch.ORIGINAL_MODEL(42) if kind=='resnet' else dino(ROOT).cuda()).eval().requires_grad_(False)
  assert all(not p.requires_grad for p in model.parameters()) and all(not m.training for m in model.modules())
  before=state_sha(model);holder={}
  if kind=='resnet':handle=model.layer4.register_forward_hook(lambda module,args,value:holder.update(feature=value))
  count=0
  with torch.inference_mode():
   for x,_,_,idx in core.loader(list(range(2386)),False,seed=42):
    x=x.cuda(non_blocking=True)
    if kind=='resnet':model(x);feature[idx.numpy(),:512]=holder['feature'].float().mean((2,3)).cpu().numpy()
    else:
     tokens=model.forward_features(x);assert tokens.shape[1:]==(257,384)
     feature[idx.numpy(),512:896]=tokens[:,0].cpu().numpy();feature[idx.numpy(),896:]=tokens[:,1:].mean(1).cpu().numpy()
    count+=len(idx)
  if kind=='resnet':handle.remove()
  after=state_sha(model);assert before==after and count==2386
  details[kind]=dict(state_sha256=after,parameters=sum(p.numel() for p in model.parameters()),
                     feature_parameters=sum(p.numel() for n,p in model.named_parameters() if kind!='resnet' or not n.startswith('fc.')))
  del model;torch.cuda.empty_cache();print('extracted independent encoder',kind,flush=True)
 assert np.isfinite(feature).all()
 temporary=path.with_suffix('.tmp')
 with temporary.open('wb') as f:np.savez(f,features=feature,indices=np.arange(2386))
 os.replace(temporary,path)
 c=dict(signature=signature,sha256=core.digest(path),shape=list(feature.shape),indices=list(range(2386)),
        encoders=details,new_neural_trainings=0,precision='float32, TF32 disabled',seconds=time.time()-started)
 core.save_json(meta,c);return feature,c

def worker(job_id):
 from sklearn.linear_model import Ridge
 verify();job=next(j for j in core.read_json(STUDY/'final_jobs.json') if j['id']==job_id)
 if done(job):return
 feature,cache=extract();records=core.read_json(STUDY/'manifest.json');y=np.array([r['label'] for r in records[:2386]])
 out=STUDY/'runs'/job_id;out.mkdir(parents=True,exist_ok=True);folder=HEADS/job_id;folder.mkdir(parents=True,exist_ok=True)
 results=[];artifacts={}
 for kind in NEW:
  x,columns,mean,scale=transform(feature,job['fit'],kind)
  head=Ridge(alpha=1.,fit_intercept=True,solver='cholesky').fit(x[job['fit']],np.eye(3)[y[job['fit']]])
  scores=head.predict(x[job['score']]);scores-=scores.max(1,keepdims=True);p=np.exp(scores);p/=p.sum(1,keepdims=True)
  path=folder/(kind+'.npz');np.savez(path,columns=columns,mean=mean,scale=scale,coef=head.coef_,intercept=head.intercept_,fit_indices=np.array(job['fit']))
  pred=out/(kind+'_outer_predictions.csv');core.save_csv(pred,core.prediction_rows(p,y[job['score']],job['score']))
  results.append(dict(scheme=kind,seed=job['seed'],fold=job['fold'],n=len(job['score']),head=str(path.relative_to(ROOT)),head_sha256=core.digest(path),dimensions=len(columns),**core.metrics(p,y[job['score']])))
  artifacts[kind]=dict(head=str(path.relative_to(ROOT)),head_sha256=core.digest(path),prediction_sha256=core.digest(pred))
 core.save_json(out/'results.json',results);core.save_json(out/'complete.json',dict(signature=core.signature(job),cache_sha256=cache['sha256'],
    artifacts=artifacts,results_sha256=core.digest(out/'results.json'),new_neural_trainings=0,ridge_fits=4,completed_at=core.timestamp()))

def done(job):
 out=STUDY/'runs'/job['id'];path=out/'complete.json'
 if not path.exists():return False
 c=core.read_json(path);assert c['signature']==core.signature(job) and core.digest(out/'results.json')==c['results_sha256']
 assert core.digest(CACHE/'features.npz')==c['cache_sha256']
 for name,a in c['artifacts'].items():
  assert core.digest(ROOT/a['head'])==a['head_sha256'] and core.digest(out/(name+'_outer_predictions.csv'))==a['prediction_sha256']
 return True
arch.done=done;arch.baseline_predictions=lambda:None

def check():
 import torch
 import torch.nn.functional as F
 from sklearn.linear_model import Ridge
 cfg=verify();records=core.read_json(STUDY/'manifest.json');jobs=core.read_json(STUDY/'final_jobs.json')
 assert len(jobs)==15
 assert core.digest(core.CACHE/'resized_uint8.npy')==core.read_json(core.CACHE/'meta.json')['cache_sha256']
 original=core.read_json(DATA/'oof_jobs.json');coverage={seed:[] for seed in [42,43,44]}
 for job in jobs:
  source=next(j for j in original if j['id']==job['source_oof_job_id'])
  for key,name in [('fit','train'),('inner','select'),('score','score')]:assert job[key]==source[name]
  core.validate_split(records,job['fit'],job['inner'],job['score']);assert all(0<=i<2386 for k in ['fit','inner','score'] for i in job[k])
  coverage[job['seed']]+=job['score']
 assert all(sorted(v)==list(range(2386)) for v in coverage.values())
 for path in cfg['reference_hashes']:
  rows=arch.read_csv(STUDY/path);assert [int(r['index']) for r in rows]==list(range(2386))
 model=dino(ROOT).cuda().eval();raw=torch.load(ROOT/'weights/pretrained/dinov2_vits14_pretrain.pth',map_location='cpu',weights_only=True)
 position=raw['pos_embed'];patch=position[:,1:].reshape(1,37,37,384).permute(0,3,1,2)
 patch=F.interpolate(patch,size=(16,16),mode='bicubic',align_corners=False,antialias=False).permute(0,2,3,1).reshape(1,256,384)
 assert torch.allclose(model.pos_embed.cpu(),torch.cat([position[:,:1],patch],1),atol=1e-6)
 x,_,_,_=next(iter(core.loader([0,1],False,seed=42)));x=x.cuda()
 with torch.inference_mode():
  tokens=model.forward_features(x);assert tokens.shape==(2,257,384) and torch.allclose(model(x),tokens[:,0],atol=1e-6)
 assert all(not p.requires_grad for p in model.parameters());del model
 feature,meta=extract()
 rng=np.random.default_rng(42);fake=rng.standard_normal((24,1280))
 for kind in NEW:
  a,cols,mean,scale=transform(fake,list(range(16)),kind)
  changed=fake.copy();changed[16:]+=100;_,_,mean2,scale2=transform(changed,list(range(16)),kind)
  assert np.array_equal(mean,mean2) and np.array_equal(scale,scale2)
  target=rng.standard_normal((16,3));head=Ridge(alpha=1.,fit_intercept=True,solver='cholesky').fit(a[:16],target)
  centered=a[:16]-a[:16].mean(0);response=target-target.mean(0)
  assert np.allclose(head.coef_.T,centered.T@np.linalg.solve(centered@centered.T+np.eye(16),response),atol=1e-8)
 # Re-extract a tiny deterministic subset, independently of the global cache loop.
 for name in ['resnet','dino']:
  m=(arch.ORIGINAL_MODEL(42) if name=='resnet' else dino(ROOT).cuda()).eval().requires_grad_(False)
  held={}
  if name=='resnet':handle=m.layer4.register_forward_hook(lambda module,args,value:held.update(feature=value))
  with torch.inference_mode():
   if name=='resnet':m(x);again=held['feature'].mean((2,3)).cpu().numpy();expected=feature[:2,:512]
   else:t=m.forward_features(x);again=torch.cat([t[:,0],t[:,1:].mean(1)],1).cpu().numpy();expected=feature[:2,512:]
  assert np.allclose(again,expected,rtol=1e-4,atol=3e-4)
  if name=='resnet':handle.remove()
  del m
 core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),encoders=meta['encoders'],checks=[
   'public weight hashes and strict all-key DINO inference loading','declared positional interpolation independently reproduced',
   'token shape and CLS forward agreement','all frozen model states unchanged by complete FP32 inference',
   'only original-train indices and original three-way patient splits, complete coverage','copied six OOF references locked',
   'deterministic subset features reproduced','fit-only transforms and independent ridge closed form','zero neural training']))
 arch.commit('test: validate independent pretrained encoders, frozen features and fit-only probes');print('PASS pretrained representation preflight',flush=True)

def summarize():
 cfg=verify();jobs=core.read_json(STUDY/'final_jobs.json');preds={};summary=[];folds=[]
 for scheme in ALL:
  for seed in [42,43,44]:
   rows=[]
   if scheme in NEW:
    for job in [j for j in jobs if j['seed']==seed]:
     assert done(job);out=STUDY/'runs'/job['id'];rows+=arch.read_csv(out/(scheme+'_outer_predictions.csv'))
     folds+=[r for r in core.read_json(out/'results.json') if r['scheme']==scheme]
   else:rows=arch.read_csv(STUDY/'references'/f'{scheme}_s{seed}.csv')
   rows=sorted(rows,key=lambda r:int(r['index']));assert [int(r['index']) for r in rows]==list(range(2386))
   preds[scheme,seed]=rows;core.save_csv(STUDY/'oof'/f'{scheme}_s{seed}.csv',rows)
   p=np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows]);y=np.array([int(r['label']) for r in rows]);metrics=core.metrics(p,y)
   groups=sorted({r['group_id'] for r in rows});cm=np.array(metrics['confusion_matrix'])
   accuracy=float(np.mean([np.mean([r['pred']==r['label'] for r in rows if r['group_id']==g]) for g in groups]))
   summary.append(dict(scheme=scheme,seed=seed,n=2386,malignant_recall=float(cm[1,1]/cm[1].sum()),patient_mean_accuracy=accuracy,**metrics))
 weights=np.random.default_rng(20261024).multinomial(185,np.ones(185)/185,size=20000)
 def matrix(rows):
  m=np.zeros((185,3,3));np.add.at(m,([groups.index(r['group_id']) for r in rows],[int(r['label']) for r in rows],[int(r['pred']) for r in rows]),1);return m
 def f1(m):
  d=m.sum(-1)+m.sum(-2);return np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),d,out=np.zeros_like(d),where=d!=0).mean(-1)
 pairs=[(s,'resnet_frozen_gap') for s in ['dino_cls','dino_mean','dino_concat']]+[('dino_concat',s) for s in REFS];comparisons={}
 for candidate,baseline in pairs:
  deltas=[];boots=[]
  for seed in [42,43,44]:
   a=matrix(preds[baseline,seed]);b=matrix(preds[candidate,seed]);deltas.append(float(f1(b.sum(0))-f1(a.sum(0))))
   boots.append(f1(np.einsum('bg,gij->bij',weights,b))-f1(np.einsum('bg,gij->bij',weights,a)))
  boot=np.mean(boots,axis=0);family=np.percentile(boot,[.5,99.5]).tolist()
  comparisons[candidate+'_vs_'+baseline]=dict(seed_deltas=deltas,mean_delta=float(np.mean(deltas)),patient95=np.percentile(boot,[2.5,97.5]).tolist(),family_interval=family,
     strong_pair=min(deltas)>0 and np.mean(deltas)>=.01 and family[0]>0)
 strong=all(comparisons['dino_concat_vs_'+b]['strong_pair'] for b in ['resnet_frozen_gap']+REFS)
 core.save_json(STUDY/'round9_results.json',summary);core.save_json(STUDY/'fold_results.json',folds)
 core.save_json(STUDY/'paired_comparisons.json',dict(comparisons=comparisons,strong_primary_followup=strong))
 lines=['# Round9 frozen pretrained representation baselines','','Zero neural training;60 new ridge fits;6 copied OOF reference predictions. Pretraining/architecture/precision/compute differ; not a new network.','', '| Scheme | Mean OOF F1 |','|---|---:|']
 for name in ALL:lines.append(f'| {name} | {np.mean([r["macro_f1"] for r in summary if r["scheme"]==name]):.6f} |')
 lines+=['',json.dumps(dict(strong_primary_followup=strong,comparisons=comparisons),indent=2)]
 (STUDY/'ROUND9_REPORT.md').write_text('\n'.join(lines));arch.commit('results: compare independent-pretraining representation baselines')
 core.save_json(STUDY/'status.json',dict(phase='round9_complete_needs_analysis',round=9,total=15,completed=15,new_neural_trainings=0,ridge_fits=60,updated_at=core.timestamp()))
arch.summarize=summarize

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','worker','run']);parser.add_argument('--job');args=parser.parse_args()
 if args.command=='prepare':prepare()
 elif args.command=='check':check()
 elif args.command=='worker':worker(args.job)
 else:arch.run()
