"""Independent completion audit of round 6; no training or model inference."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_transfer_study as study
core=study.core
cfg=study.verify()
records=core.read_json(study.STUDY/'manifest.json')
jobs=core.read_json(study.STUDY/'final_jobs.json')
assert len(jobs)==12

def rows(path):
 with open(path) as f:return list(csv.DictReader(f))

def check_predictions(out,result):
 pred=rows(out/'validation_predictions.csv')
 assert len(pred)==538 and {int(r['index']) for r in pred}==set(cfg['val_indices'])
 p=np.array([[float(r[f'p{k}']) for k in range(3)] for r in pred])
 y=np.array([int(r['label']) for r in pred])
 assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=2e-6)
 assert np.array_equal(p.argmax(1),[int(r['pred']) for r in pred])
 for r in pred:
  m=records[int(r['index'])]
  assert r['sample_id']==m['sample_id'] and r['group_id']==m['group_id'] and int(r['label'])==m['label']
 metrics=core.metrics(p,y)
 assert abs(metrics['macro_f1']-result['macro_f1'])<1e-12
 assert metrics['confusion_matrix']==result['confusion_matrix']
 assert all(p.owner()=='login' for p in out.rglob('*') if p.is_file())
 return pred


from transfer_policies import configure
canonical=study.arch.ORIGINAL_MODEL(42).cpu()
initial={k:v.detach().clone() for k,v in canonical.state_dict().items()}
bn_names=[name for name,m in canonical.named_modules() if isinstance(m,torch.nn.BatchNorm2d)]
audit=[];curves=[];predictions={};frozen_checks=[]
for job in jobs:
 assert study.arch.done(job)
 out=study.STUDY/'runs'/job['id']
 comp=core.read_json(out/'complete.json');result=core.read_json(out/'validation_complete.json')
 assert core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']==result['checkpoint_sha256']
 predictions[job['variant'],job['seed']]=check_predictions(out,result)
 log=rows(out/'train_log.csv')
 assert [int(r['epoch']) for r in log]==list(range(60))
 assert all(np.isfinite(float(v)) for r in log for v in r.values())
 assert abs(max(float(r['val_macro_f1']) for r in log)-result['macro_f1'])<1e-12
 assert int(comp['best_epoch'])==int(max(log,key=lambda r:float(r['val_macro_f1']))['epoch'])
 assert set(job['score'])==set(job['select'])==set(cfg['val_indices'])
 assert set(job['train'])==set(range(2386))
 assert {records[i]['split'] for i in job['train']}=={'train'}
 assert not set(job['train']) & set(cfg['test_indices'])
 assert not {records[i]['group_id'] for i in job['train']} & {records[i]['group_id'] for i in job['score']}
 assert (ROOT/comp['checkpoint']).owner()=='login'
 model=canonical
 configure(model,job['variant'])
 state=torch.load(ROOT/comp['checkpoint'],map_location='cpu',weights_only=True)
 frozen_params=[];changed_trainable=[];squares=0.;count=0
 for name,param in model.named_parameters():
  assert torch.isfinite(state[name]).all()
  if not param.requires_grad:
   assert torch.equal(state[name],initial[name]),name
   frozen_params.append(name)
  elif not torch.equal(state[name],initial[name]):changed_trainable.append(name)
  if not name.startswith('fc.'):
   squares+=float((state[name]-initial[name]).double().square().sum());count+=param.numel()
 assert 'fc.weight' in changed_trainable
 trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
 assert trainable==comp['trainable_parameters']==result['trainable_parameters']
 frozen_bn=[];active_bn=[]
 for name in bn_names:
  frozen=job['variant'] in {'head_only','bn_frozen'} or (job['variant']=='stage4_only' and not name.startswith('layer4.'))
  for suffix in ['running_mean','running_var','num_batches_tracked']:
   key=name+'.'+suffix;assert torch.isfinite(state[key]).all()
   if frozen:assert torch.equal(state[key],initial[key]),key
  if not frozen:assert state[name+'.num_batches_tracked']>initial[name+'.num_batches_tracked']
  (frozen_bn if frozen else active_bn).append(name)
 rms=(squares/count)**.5
 assert abs(rms-result['diagnostics']['backbone_rms_drift'])<1e-8
 if job['variant']=='starting_anchor':assert max(float(r['anchor_penalty']) for r in log)>0
 frozen_checks.append(dict(job=job['id'],frozen_parameter_tensors=len(frozen_params),changed_trainable_tensors=len(changed_trainable),
                           trainable_parameters=trainable,frozen_bn=len(frozen_bn),active_bn=len(active_bn),backbone_rms_drift=rms))
 curves.append(dict(variant=job['variant'],seed=job['seed'],best_epoch=comp['best_epoch'],
                    final_train_accuracy=float(log[-1]['train_acc']),final_train_ce=float(log[-1]['train_ce']),final_anchor_penalty=float(log[-1]['anchor_penalty']),
                    final_backbone_rms_drift=float(log[-1]['backbone_rms_drift']),
                    final_validation_f1=float(log[-1]['val_macro_f1']),best_validation_f1=result['macro_f1']))
 audit.append(dict(job=job['id'],epochs=60,validation_images=538,checkpoint_sha256=comp['checkpoint_sha256']))

baselines=[];original=core.read_json(study.DATA/'final_jobs.json')
for arm,recipe in [('full_ce','full'),('grade_ce','grade_weighted')]:
 for seed in cfg['seeds']:
  job=next(j for j in original if j['arm']==recipe and j['seed']==seed)
  comp=core.read_json(study.DATA/'runs'/job['id']/'complete.json')
  sig=hashlib.sha256((json.dumps(job,sort_keys=True)+core.digest(study.DATA/'config.json')+core.digest(core.__file__)).encode()).hexdigest()
  out=study.STUDY/'baselines'/f'{arm}_s{seed}';result=core.read_json(out/'validation_result.json')
  assert comp['signature']==sig==result['source_signature']
  assert core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']==result['checkpoint_sha256']
  assert comp['epochs_completed']==60 and result['reused'] and (ROOT/comp['checkpoint']).owner()=='login'
  log=rows(study.DATA/'runs'/job['id']/'train_log.csv')
  assert [int(r['epoch']) for r in log]==list(range(60))
  predictions[arm,seed]=check_predictions(out,result)
  assert abs(result['macro_f1']-comp['best_select_f1'])<1e-12
  baselines.append(dict(variant=arm,seed=seed,source_signature=sig,checkpoint_sha256=comp['checkpoint_sha256']))

errors={}
for arm in study.ALL_ARMS:
 rr=[predictions[arm,s] for s in cfg['seeds']]
 cm=np.array([core.metrics(np.array([[float(r[f'p{k}']) for k in range(3)] for r in v]),np.array([int(r['label']) for r in v]))['confusion_matrix'] for v in rr])
 errors[arm]=dict(mean_confusion_matrix=cm.mean(0).tolist(),wrong_all_seeds=sum(all(v[i]['pred']!=v[i]['label'] for v in rr) for i in range(538)))
cross_method=[]
for pos,r in enumerate(predictions['full_ce',42]):
 votes=sum(v[pos]['pred']!=v[pos]['label'] for v in predictions.values())
 if votes>=15:cross_method.append(dict(index=int(r['index']),sample_id=r['sample_id'],group_id=r['group_id'],label=int(r['label']),wrong_votes=votes))
core.save_json(study.STUDY/'training_and_error_diagnostics.json',dict(curves=curves,errors=errors,
 persistent_errors_15_of_18=cross_method,limitations=['Exploratory same-validation diagnostics; no new clinical truth or automatic relabeling.','Objectives have different training losses; only CE and accuracy are comparable across arms.']))
core.save_json(study.STUDY/'completion_audit.json',dict(passed=True,timestamp=core.timestamp(),new_trainings=12,reused_models=6,
 checks=['immutable sources/config and baseline files','12 new and 6 reused checkpoint SHA256 and signatures','60 epochs each',
         '18 complete validation sets of 538 images','recomputed probabilities and metrics','patient isolation, no test scoring',
         'frozen parameter equality to pretrained initialization','frozen BN statistics and counters unchanged, active counters incremented',
         'trainable counts and RMS parameter drift','login ownership'],jobs=audit,baselines=baselines,policies=frozen_checks))
print(json.dumps(dict(passed=True,new_trainings=12,reused_models=6,policies=frozen_checks,curves=curves),indent=2))
