#!/usr/bin/env python3
"""Round 7: paired patient OOF normalization check, reusing 15 valid baselines."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_architecture_study as arch
core=arch.core
DATA=ROOT/'experiments/data_study'
STUDY=ROOT/'experiments/bn_oof_study'
arch.STUDY=STUDY;arch.__file__=__file__
core.STUDY=STUDY;core.WEIGHTS=ROOT/'weights/bn_oof_study'
ORIGINAL_MODEL=arch.ORIGINAL_MODEL
ORIGINAL_SAVE=core.save_json

def save_json(path,value):
 if Path(path)==STUDY/'status.json' and value.get('phase')=='round1_training':
  value=dict(value,phase='round7_training',round=7,reused_baseline_models=15)
 ORIGINAL_SAVE(path,value)
core.save_json=save_json

def frozen_model(seed):
 import torch
 model=ORIGINAL_MODEL(seed)
 original_train=model.train
 def mode(self,training=True):
  original_train(training)
  if training:
   for m in self.modules():
    if isinstance(m,torch.nn.BatchNorm2d):m.eval()
  return self
 model.train=types.MethodType(mode,model)
 return model
core.new_model=frozen_model

def sources():
 return dict(driver=core.digest(__file__),protocol=core.digest(ROOT/'experiments/ROUND7_PROTOCOL.md'),
             trainer=core.digest(ROOT/'experiments/run_data_study.py'),
             supervisor=core.digest(ROOT/'experiments/run_architecture_study.py'),
             backbone=core.digest(ROOT/'cle_fair_common.py'))

def verify():
 cfg=core.read_json(STUDY/'config.json')
 assert cfg['architecture_sources']==sources()
 assert core.digest(STUDY/'manifest.json')==cfg['manifest_sha256']
 assert core.digest(DATA/'config.json')==cfg['source_data_config_sha256']
 assert core.digest(DATA/'oof_jobs.json')==cfg['source_oof_jobs_sha256']
 assert core.digest(core.CACHE/'meta.json')==cfg['cache_meta_sha256']
 for path,sha in cfg['baseline_file_hashes'].items():assert core.digest(STUDY/path)==sha
 return cfg
arch.verify=verify

def source_signature(job):
 return hashlib.sha256((json.dumps(job,sort_keys=True)+core.digest(DATA/'config.json')+core.digest(core.__file__)).encode()).hexdigest()

def prepare():
 STUDY.mkdir(parents=True,exist_ok=True)
 if (STUDY/'config.json').exists():verify();return
 records=core.read_json(DATA/'manifest.json');original=core.read_json(DATA/'oof_jobs.json')
 assert len(original)==15
 (STUDY/'manifest.json').write_bytes((DATA/'manifest.json').read_bytes())
 cfg=core.read_json(DATA/'config.json')
 for k in ['grade_rules','grade_weights','confidence','weight_normalization','arms']:cfg.pop(k,None)
 cfg.update(round=7,created_at=core.timestamp(),architecture_sources=sources(),
     source_data_config_sha256=core.digest(DATA/'config.json'),source_oof_jobs_sha256=core.digest(DATA/'oof_jobs.json'),
     cache_meta_sha256=core.digest(core.CACHE/'meta.json'),new_trainings=15,reused_models=15,
     variants=['bn_train','bn_frozen'],grading='Not used; sample weights all 1.',
     selection='Existing inner patient selection only; outer patients never select checkpoint.',
     active_indices=list(range(2386)),validation_policy='Original fixed val excluded from training, selection and scoring.',
     test_policy='No test evaluation, label changes or filtering.',
     limitation='Internal patient OOF check of development-selected weak signal, not independent external confirmation.')
 hashes={};index=[];jobs=[]
 for old in original:
  assert old['phase']=='oof' and not old.get('sample_weights')
  comp=core.read_json(DATA/'runs'/old['id']/'complete.json')
  assert comp['signature']==source_signature(old) and comp['epochs_completed']==60
  assert core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']
  out=STUDY/'baselines'/old['id'];out.mkdir(parents=True,exist_ok=True)
  for name in ['complete.json','run_meta.json','train_log.csv','oof_predictions.csv']:
   source=DATA/'runs'/old['id']/name;target=out/name;shutil.copyfile(source,target)
   hashes[str(target.relative_to(STUDY))]=core.digest(target)
  index.append(dict(source_job=old,checkpoint_sha256=comp['checkpoint_sha256'],source_signature=comp['signature']))
  job=dict(old);job.update(id=f'r7_bn_frozen_s{old["seed"]}_f{old["fold"]}',arm='bn_frozen',variant='bn_frozen',
                           source_job_id=old['id'],architecture_sources=cfg['architecture_sources'])
  jobs.append(job)
 cfg['baseline_file_hashes']=hashes
 core.save_json(STUDY/'config.json',cfg);core.save_json(STUDY/'baseline_index.json',index)
 core.save_json(STUDY/'final_jobs.json',jobs)
 core.save_json(STUDY/'status.json',dict(phase='prepared',round=7,total=15,reused_baseline_models=15,updated_at=core.timestamp()))
 arch.commit('research: freeze paired patient OOF BN check and reuse 15 signed baselines')

def validate_predictions(path,ids,records):
 import numpy as np
 rows=sorted(arch.read_csv(path),key=lambda r:int(r['index']))
 assert [int(r['index']) for r in rows]==sorted(ids)
 p=np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows]);y=np.array([int(r['label']) for r in rows])
 assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=2e-6)
 for r in rows:
  m=records[int(r['index'])]
  assert m['split']=='train' and m['sample_id']==r['sample_id'] and m['group_id']==r['group_id'] and m['label']==int(r['label'])
 assert np.array_equal(p.argmax(1),[int(r['pred']) for r in rows])
 return rows,core.metrics(p,y)

def worker(job_id):
 import torch
 cfg=verify();job=next(j for j in core.read_json(STUDY/'final_jobs.json') if j['id']==job_id)
 if not core.complete(job):core.run_worker(job_id)
 out=STUDY/'runs'/job_id;comp=core.read_json(out/'complete.json');path=out/'oof_predictions.csv'
 if not path.exists():
  model=frozen_model(job['seed']);model.load_state_dict(torch.load(ROOT/comp['checkpoint'],map_location='cpu',weights_only=True),strict=True)
  p,y,idx=core.infer(model,core.loader(job['score'],False,seed=job['seed']))
  core.save_csv(path,core.prediction_rows(p,y,idx))
 records=core.read_json(STUDY/'manifest.json');rows,metrics=validate_predictions(path,job['score'],records)
 assert abs(metrics['macro_f1']-comp['outer_metrics']['macro_f1'])<1e-12
 core.save_json(out/'outer_complete.json',dict(signature=core.signature(job),n=len(rows),
     checkpoint_sha256=comp['checkpoint_sha256'],prediction_sha256=core.digest(path),completed_at=core.timestamp(),**metrics))

def done(job):
 path=STUDY/'runs'/job['id']/'outer_complete.json'
 if not path.exists():return False
 assert core.complete(job)
 result=core.read_json(path)
 assert result['signature']==core.signature(job) and result['n']==len(job['score'])
 assert core.digest(path.parent/'oof_predictions.csv')==result['prediction_sha256']
 return True
arch.done=done;arch.baseline_predictions=lambda:None

def check():
 import numpy as np
 import torch
 import torch.nn.functional as F
 cfg=verify();records=core.read_json(STUDY/'manifest.json')
 assert core.digest(core.PRETRAINED)==cfg['pretrained_sha256']
 meta=core.read_json(core.CACHE/'meta.json')
 assert core.digest(core.CACHE/'resized_uint8.npy')==meta['cache_sha256']
 coverage=collections.Counter();baseline=[]
 newjobs=core.read_json(STUDY/'final_jobs.json')
 for entry in core.read_json(STUDY/'baseline_index.json'):
  old=entry['source_job'];job=next(j for j in newjobs if j['source_job_id']==old['id'])
  for k in ['train','select','score','seed','fold','epochs']:assert job[k]==old[k]
  assert all(0<=i<2386 for k in ['train','select','score'] for i in job[k])
  core.validate_split(records,job['train'],job['select'],job['score'])
  coverage.update((job['seed'],i) for i in job['score'])
  out=STUDY/'baselines'/old['id'];comp=core.read_json(out/'complete.json')
  assert comp['signature']==source_signature(old) and core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']
  assert [int(r['epoch']) for r in arch.read_csv(out/'train_log.csv')]==list(range(60))
  rr,metrics=validate_predictions(out/'oof_predictions.csv',old['score'],records)
  assert abs(metrics['macro_f1']-comp['outer_metrics']['macro_f1'])<1e-12
  baseline.append(dict(id=old['id'],outer_n=len(rr),outer_macro_f1=metrics['macro_f1'],checkpoint_sha256=comp['checkpoint_sha256']))
 assert set(coverage)=={(s,i) for s in [42,43,44] for i in range(2386)} and set(coverage.values())=={1}
 job=newjobs[0];x,y,_,_=next(iter(core.loader(job['train'][:32],False,seed=42)));x=x.cuda();y=y.cuda()
 expected=ORIGINAL_MODEL(42).eval();m=frozen_model(42).eval()
 assert list(m.state_dict())==list(expected.state_dict())
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):assert torch.equal(m(x),expected(x))
 initial={k:v.detach().clone() for k,v in m.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
 m.train();bns=[v for v in m.modules() if isinstance(v,torch.nn.BatchNorm2d)]
 assert len(bns)==36 and all(not v.training for v in bns) and m.training
 assert all(p.requires_grad for p in m.parameters())
 optimizer=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.05);scaler=torch.amp.GradScaler('cuda',init_scale=128.)
 with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(m(x),y)
 scaler.scale(loss).backward();assert torch.isfinite(m.fc.weight.grad).all() and m.fc.weight.grad.abs().sum()>0
 assert m.layer4[-1].bn2.weight.grad.abs().sum()>0
 scaler.step(optimizer);scaler.update()
 assert all(torch.equal(m.state_dict()[k],v) for k,v in initial.items())
 m.eval();assert all(not v.training for v in m.modules())
 core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),baselines=baseline,
    checks=['15 original signed checkpoint hashes, 60 epochs and outer predictions','paired original outer/inner splits unchanged',
            'patient isolation, all classes and only original-train indices','full cache SHA256, index-restricted access',
            'exact OOF coverage 2386 per seed','paired eval logits and state keys identical','all parameters trainable',
            'frozen BN buffers remain unchanged during finite AMP update, affine gradient nonzero','eval disables all submodules',
            'exact immutable core.run_worker reused; no grades or weights']))
 arch.commit('test: validate reused OOF folds and trainable-affine frozen-statistics BN')
 print(json.dumps(dict(passed=True,new_models=15,reused_baselines=15,train_images=2386,train_groups=185)),flush=True)

def summarize():
 import numpy as np
 cfg=verify();records=core.read_json(STUDY/'manifest.json');jobs=core.read_json(STUDY/'final_jobs.json')
 predictions={};folds=[];summary=[]
 for arm in ['bn_train','bn_frozen']:
  for seed in cfg['seeds']:
   merged=[]
   for job in [j for j in jobs if j['seed']==seed]:
    if arm=='bn_train':out=STUDY/'baselines'/job['source_job_id']
    else:assert done(job);out=STUDY/'runs'/job['id']
    rr,metrics=validate_predictions(out/'oof_predictions.csv',job['score'],records);merged+=rr
    folds.append(dict(variant=arm,seed=seed,fold=job['fold'],n=len(rr),**metrics))
   merged=sorted(merged,key=lambda r:int(r['index']))
   assert [int(r['index']) for r in merged]==list(range(2386))
   predictions[arm,seed]=merged
   p=np.array([[float(r[f'p{k}']) for k in range(3)] for r in merged]);y=np.array([int(r['label']) for r in merged])
   core.save_csv(STUDY/'oof'/f'{arm}_s{seed}.csv',merged)
   metrics=core.metrics(p,y);groups=sorted({r['group_id'] for r in merged})
   acc=np.mean([np.mean([r['pred']==r['label'] for r in merged if r['group_id']==g]) for g in groups])
   cm=np.array(metrics['confusion_matrix'])
   summary.append(dict(variant=arm,seed=seed,n=2386,patient_groups=len(groups),malignant_recall=float(cm[1,1]/cm[1].sum()),patient_mean_accuracy=float(acc),**metrics))
 bw=np.random.default_rng(20261024).multinomial(len(groups),np.ones(len(groups))/len(groups),size=20000)
 def matrix(rr):
  m=np.zeros((len(groups),3,3));np.add.at(m,([groups.index(r['group_id']) for r in rr],[int(r['label']) for r in rr],[int(r['pred']) for r in rr]),1);return m
 def f1(m):
  d=m.sum(-1)+m.sum(-2);return np.mean(np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),d,out=np.zeros_like(d),where=d!=0),axis=-1)
 deltas=[];boot=[]
 for seed in cfg['seeds']:
  a=matrix(predictions['bn_train',seed]);b=matrix(predictions['bn_frozen',seed])
  deltas.append(float(f1(b.sum(0))-f1(a.sum(0))))
  boot.append(f1(np.einsum('bg,gij->bij',bw,b))-f1(np.einsum('bg,gij->bij',bw,a)))
 comparison=dict(seed_deltas=deltas,mean_delta=float(np.mean(deltas)),patient_cluster95=np.percentile(np.mean(boot,axis=0),[2.5,97.5]).tolist(),patient_groups=len(groups),resamples=20000)
 comparison['strong_diagnostic_signal']=min(deltas)>0 and comparison['mean_delta']>=.01 and comparison['patient_cluster95'][0]>0
 core.save_json(STUDY/'round7_results.json',summary);core.save_json(STUDY/'fold_results.json',folds);core.save_json(STUDY/'paired_comparison.json',comparison)
 lines=['# Round 7: internal patient OOF normalization check','','15 new frozen-BN trainings, 15 reused standard-BN models. No grades, fixed val or test used. Internal check, not external confirmation.','', '| Arm | Seed | Pooled OOF macro-F1 | Malignant recall | Patient mean accuracy |','|---|---:|---:|---:|---:|']
 for r in summary:lines.append(f'| {r["variant"]} | {r["seed"]} | {r["macro_f1"]:.6f} | {r["malignant_recall"]:.6f} | {r["patient_mean_accuracy"]:.6f} |')
 lines+=['',json.dumps(comparison,indent=2),'','Intervals condition on cross-fitted models; overlapping fold-training sets, development selection and training uncertainty limit interpretation.']
 (STUDY/'ROUND7_REPORT.md').write_text('\n'.join(lines))
 arch.commit('results: compare frozen BN and standard BN on paired patient OOF predictions')
 core.save_json(STUDY/'status.json',dict(phase='round7_complete_needs_analysis',round=7,total=15,completed=15,reused_baseline_models=15,updated_at=core.timestamp()))
arch.summarize=summarize

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','worker','run']);parser.add_argument('--job');args=parser.parse_args()
 if args.command=='prepare':prepare()
 elif args.command=='check':check()
 elif args.command=='worker':worker(args.job)
 else:arch.run()
