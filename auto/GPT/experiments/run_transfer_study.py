#!/usr/bin/env python3
"""Round 6: constrain pretrained adaptation, unchanged inference architecture."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_architecture_study as arch
core=arch.core
DATA=ROOT/'experiments/data_study'
STUDY=ROOT/'experiments/transfer_study'
OBJECTIVES=ROOT/'experiments/transfer_policies.py'
NEW_ARMS=['head_only','stage4_only','bn_frozen','starting_anchor']
ALL_ARMS=['full_ce','grade_ce']+NEW_ARMS
arch.STUDY=STUDY;arch.MODEL=OBJECTIVES;arch.__file__=__file__
core.STUDY=STUDY;core.WEIGHTS=ROOT/'weights/transfer_study'
ORIGINAL_VERIFY=arch.verify;ORIGINAL_DONE=arch.done;ORIGINAL_SAVE=core.save_json


def save_json(path,value):
 if Path(path)==STUDY/'status.json' and value.get('phase')=='round1_training':
  value=dict(value,phase='round6_training',round=6,reused_baseline_models=6)
 ORIGINAL_SAVE(path,value)


core.save_json=save_json


def sources():
 return dict(driver=core.digest(__file__),objectives=core.digest(OBJECTIVES),
             protocol=core.digest(ROOT/'experiments/ROUND6_PROTOCOL.md'),
             trainer=core.digest(ROOT/'experiments/run_data_study.py'),
             supervisor=core.digest(ROOT/'experiments/run_architecture_study.py'),
             backbone=core.digest(ROOT/'cle_fair_common.py'))


def verify():
 cfg=ORIGINAL_VERIFY()
 for path,sha in cfg['baseline_file_hashes'].items():assert core.digest(STUDY/path)==sha
 return cfg


arch.sources=sources;arch.verify=verify


def policy_reference(model):
 from transfer_policies import reference
 return reference(model)


def baseline_files(cfg):
 hashes={};reference=ROOT/'experiments/robust_study'
 oldcfg=core.read_json(reference/'config.json')
 for relative,sha in oldcfg['baseline_file_hashes'].items():
  source=reference/relative;assert core.digest(source)==sha
  target=STUDY/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
  hashes[relative]=core.digest(target)
 original=core.read_json(DATA/'final_jobs.json')
 for arm,recipe in [('full_ce','full'),('grade_ce','grade_weighted')]:
  for seed in cfg['seeds']:
   job=next(j for j in original if j['arm']==recipe and j['seed']==seed)
   comp=core.read_json(DATA/'runs'/job['id']/'complete.json')
   sig=hashlib.sha256((json.dumps(job,sort_keys=True)+core.digest(DATA/'config.json')+core.digest(core.__file__)).encode()).hexdigest()
   result=core.read_json(STUDY/'baselines'/f'{arm}_s{seed}'/'validation_result.json')
   assert comp['signature']==sig==result['source_signature'] and comp['epochs_completed']==60
   assert core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']==result['checkpoint_sha256']
 return hashes


def prepare():
 STUDY.mkdir(parents=True,exist_ok=True)
 if (STUDY/'config.json').exists():verify();return
 cfg=core.read_json(DATA/'config.json')
 (STUDY/'manifest.json').write_bytes((DATA/'manifest.json').read_bytes())
 original=core.read_json(DATA/'final_jobs.json')
 cfg.update(round=6,created_at=core.timestamp(),architecture_sources=sources(),
     grade_manifest_sha256=core.digest(DATA/'train_grades.csv'),data_protocol_sha256=core.digest(DATA/'config.json'),
     variants=NEW_ARMS,comparison_arms=ALL_ARMS,new_trainings=12,reused_models=6,
     objective_hyperparameters=dict(starting_anchor_strength=.01),recipe='grade_weighted',
     primary_hypothesis='Constrain pretrained adaptation to improve patient generalization over existing full fine-tuning.',
     test_policy='No test evaluation or filtering; labels unchanged.',
     inference='Same plain ResNet34 at 224, no additional modules or metadata.',
     provenance=dict(starting_point_penalty='https://proceedings.mlr.press/v80/li18a.html',
                     implementation='Independent L2-SP-inspired addon to unchanged AdamW, not exact paper reproduction.'))
 cfg['baseline_file_hashes']=baseline_files(cfg)
 jobs=[]
 for variant in NEW_ARMS:
  for seed in cfg['seeds']:
   j=dict(next(j for j in original if j['arm']=='grade_weighted' and j['seed']==seed))
   j.update(id=f'r6_{variant}_s{seed}',phase='transfer_constraint',arm=variant,variant=variant,recipe='grade_weighted',
            score=cfg['val_indices'],architecture_sources=cfg['architecture_sources'])
   jobs.append(j)
 core.save_json(STUDY/'config.json',cfg);core.save_json(STUDY/'final_jobs.json',jobs)
 core.save_json(STUDY/'status.json',dict(phase='prepared',round=6,total=12,reused_baseline_models=6,updated_at=core.timestamp()))
 arch.commit('research: freeze pretrained-adaptation constraints and six verified reused baselines')
 print(json.dumps(dict(new_jobs=12,reused_models=6)),flush=True)


def train_job(job):
 import torch
 import torch.nn.functional as F
 from sklearn.metrics import f1_score
 from transfer_policies import configure,training_mode,reference,anchor,drift
 cfg=verify();assert core.digest(core.PRETRAINED)==cfg['pretrained_sha256']
 model=arch.ORIGINAL_MODEL(job['seed']);configure(model,job['variant']);initial=reference(model)
 out=STUDY/'runs'/job['id'];out.mkdir(parents=True,exist_ok=True)
 ckdir=core.WEIGHTS/job['id'];ckdir.mkdir(parents=True,exist_ok=True);ck=ckdir/'model_best.pth'
 weights={int(k):v for k,v in job['sample_weights'].items()}
 train=core.loader(job['train'],True,weights,job['seed']);val=core.loader(job['select'],False,seed=job['seed'])
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.05)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=job['epochs'])
 scaler=torch.amp.GradScaler('cuda');best=-1.;best_epoch=-1;history=[];started=time.time()
 meta=dict(job_id=job['id'],signature=core.signature(job),seed=job['seed'],arm=job['arm'],
           train_count=len(job['train']),select_count=len(job['select']),created_at=core.timestamp(),
           git_commit=subprocess.check_output([core.GIT,'-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
           pretrained_sha256=cfg['pretrained_sha256'],trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
           objective_hyperparameters=cfg['objective_hyperparameters'])
 core.save_json(out/'run_meta.json',meta)
 for epoch in range(job['epochs']):
  training_mode(model,job['variant']);total_loss=total_ce=total_anchor=correct=n=0
  for x,y,w,idx in train:
   x,y,w=x.cuda(non_blocking=True),y.cuda(non_blocking=True),w.cuda(non_blocking=True).float()
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):
    logits=model(x);ce=F.cross_entropy(logits,y,reduction='none');weighted=(ce*w).mean()
   penalty=anchor(model,initial) if job['variant']=='starting_anchor' else torch.zeros((),device='cuda')
   loss=weighted+penalty
   if not torch.isfinite(loss):raise RuntimeError('nonfinite training objective')
   scaler.scale(loss).backward();scaler.step(opt);scaler.update()
   total_loss+=float(loss.detach())*len(y);total_ce+=float(ce.detach().float().mean())*len(y)
   total_anchor+=float(penalty.detach())*len(y);correct+=int((logits.argmax(1)==y).sum());n+=len(y)
  p,y,_=core.infer(model,val)
  val_f1=float(f1_score(y,p.argmax(1),average='macro',labels=[0,1,2],zero_division=0));scheduler.step()
  if val_f1>best:
   best,best_epoch=val_f1,epoch;torch.save(model.state_dict(),ck.with_suffix('.tmp'));os.replace(ck.with_suffix('.tmp'),ck)
  history.append(dict(epoch=epoch,train_loss=total_loss/n,train_ce=total_ce/n,anchor_penalty=total_anchor/n,
                      train_acc=correct/n,backbone_rms_drift=drift(model,initial),val_macro_f1=val_f1,
                      lr=opt.param_groups[0]['lr'],elapsed_seconds=round(time.time()-started,2)))
  core.save_csv(out/'train_log.csv',history)
  print(f'[{job["id"]}] epoch={epoch+1}/60 train_acc={correct/n:.4f} val_f1={val_f1:.4f} best={best:.4f}',flush=True)
 meta.update(completed_at=core.timestamp(),seconds=round(time.time()-started,2),best_select_f1=best,best_epoch=best_epoch,
             checkpoint=str(ck.relative_to(ROOT)),checkpoint_sha256=core.digest(ck),epochs_completed=len(history))
 core.save_json(out/'complete.json',meta)


def worker(job_id):
 import torch
 from transfer_policies import configure,reference,drift
 verify();job=next(j for j in core.read_json(STUDY/'final_jobs.json') if j['id']==job_id)
 if not core.complete(job):train_job(job)
 out=STUDY/'runs'/job_id;comp=core.read_json(out/'complete.json')
 model=arch.ORIGINAL_MODEL(job['seed']);initial=reference(model)
 model.load_state_dict(torch.load(ROOT/comp['checkpoint'],map_location='cpu',weights_only=True),strict=True)
 configure(model,job['variant'])
 p,y,idx=core.infer(model,core.loader(job['score'],False,seed=job['seed']))
 result=core.metrics(p,y);assert abs(result['macro_f1']-comp['best_select_f1'])<1e-10
 core.save_csv(out/'validation_predictions.csv',core.prediction_rows(p,y,idx))
 core.save_json(out/'validation_complete.json',dict(job_id=job_id,signature=core.signature(job),variant=job['variant'],
      recipe='grade_weighted',seed=job['seed'],n=len(y),checkpoint_sha256=comp['checkpoint_sha256'],completed_at=core.timestamp(),
      parameters=sum(p.numel() for p in model.parameters()),modifier_parameters=0,trainable_parameters=comp['trainable_parameters'],
      diagnostics=dict(training_only=True,unchanged_inference_model=True,backbone_rms_drift=drift(model,initial)),**result))


def done(job):
 return ORIGINAL_DONE(job)


arch.done=done;arch.baseline_predictions=lambda:None


def check():
 import torch
 import torch.nn.functional as F
 from transfer_policies import configure,training_mode,reference,anchor
 cfg=verify();job=core.read_json(STUDY/'final_jobs.json')[0]
 x,y,_,_=next(iter(core.loader(job['train'][:32],False,seed=42)));x=x.cuda();y=y.cuda()
 canonical=arch.ORIGINAL_MODEL(42).eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):expected=canonical(x)
 del canonical
 checks=[]
 for kind in NEW_ARMS:
  m=arch.ORIGINAL_MODEL(42);initial=reference(m);state={k:v.detach().clone() for k,v in m.state_dict().items()}
  configure(m,kind);m.eval()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):assert torch.equal(m(x),expected)
  training_mode(m,kind)
  names=[k for k,p in m.named_parameters() if p.requires_grad]
  if kind=='head_only':assert set(names)=={'fc.weight','fc.bias'}
  elif kind=='stage4_only':assert all(k.startswith(('layer4.','fc.')) for k in names) and len(names)>2
  else:assert len(names)==len(list(m.parameters()))
  # Use a settled scale for this one-step gradient assertion. Real training keeps
  # the original dynamic scaler, which may skip initial overflowed updates.
  opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=.05);scaler=torch.amp.GradScaler('cuda',init_scale=128.)
  with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(m(x),y)
  if kind=='starting_anchor':assert anchor(m,initial).item()==0;loss=loss+anchor(m,initial)
  scaler.scale(loss).backward();assert torch.isfinite(loss) and torch.isfinite(m.fc.weight.grad).all() and m.fc.weight.grad.abs().sum()>0
  for name,p in m.named_parameters():
   if not p.requires_grad:assert p.grad is None
  if kind!='head_only':assert m.layer4[0].conv1.weight.grad.abs().sum()>0
  scaler.step(opt);scaler.update()
  for name,p in m.named_parameters():
   if not p.requires_grad:assert torch.equal(p,state[name])
  frozen_bn=[];active_bn=[]
  for name,bn in m.named_modules():
   if isinstance(bn,torch.nn.BatchNorm2d):
    unchanged=torch.equal(bn.num_batches_tracked,state[name+'.num_batches_tracked'])
    frozen=kind in {'head_only','bn_frozen'} or (kind=='stage4_only' and not name.startswith('layer4.'))
    assert unchanged==frozen
    (frozen_bn if frozen else active_bn).append(name)
  checks.append(dict(variant=kind,trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),frozen_bn=len(frozen_bn),active_bn=len(active_bn)))
  del m,opt,scaler,initial,state
 tiny=torch.nn.Linear(3,2,bias=False).cuda();initial=reference(tiny)
 assert anchor(tiny,initial).item()==0
 with torch.no_grad():tiny.weight.add_(.125)
 gradient=torch.autograd.grad(anchor(tiny,initial),tiny.weight)[0]
 assert torch.allclose(gradient,torch.full_like(gradient,.00125),atol=1e-7)
 core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),policies=checks,
    checks=['paired initialized eval predictions bitwise identical','exact trainable parameter sets','real-image finite AMP gradients and updates at settled scale 128; trainer retains original dynamic scale',
            'frozen parameters unchanged and required backbone/head gradients nonzero','frozen/active BN counters independently checked',
            'starting-point penalty zero at initialization and analytical gradient','six copied baseline hashes and checkpoint signatures']))
 arch.commit('test: validate transfer constraints, BN state and starting-point gradient')
 print(json.dumps(dict(passed=True,policies=checks)),flush=True)


def summarize():
 import numpy as np
 cfg=verify();jobs=core.read_json(STUDY/'final_jobs.json');results=[];preds={}
 for arm in ALL_ARMS:
  for seed in cfg['seeds']:
   if arm in NEW_ARMS:
    job=next(j for j in jobs if j['variant']==arm and j['seed']==seed);out=STUDY/'runs'/job['id']
    r=core.read_json(out/'validation_complete.json')
   else:out=STUDY/'baselines'/f'{arm}_s{seed}';r=core.read_json(out/'validation_result.json')
   rr=sorted(arch.read_csv(out/'validation_predictions.csv'),key=lambda r:int(r['index']))
   assert [int(r['index']) for r in rr]==sorted(cfg['val_indices'])
   preds[arm,seed]=rr;results.append(r)
 groups=sorted({r['group_id'] for r in rr})
 bw=np.random.default_rng(20261024).multinomial(len(groups),np.ones(len(groups))/len(groups),size=20000)
 def cm(rows):
  m=np.zeros((len(groups),3,3));np.add.at(m,([groups.index(r['group_id']) for r in rows],[int(r['label']) for r in rows],[int(r['pred']) for r in rows]),1);return m
 matrices={key:cm(rr) for key,rr in preds.items()}
 def f1(m):
  d=m.sum(-1)+m.sum(-2)
  return np.mean(np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),d,out=np.zeros_like(d),where=d!=0),axis=-1)
 def compare(a,b):
  ds=[];boots=[]
  for seed in cfg['seeds']:
   ma,mb=matrices[a,seed],matrices[b,seed]
   ds.append(float(f1(mb.sum(0))-f1(ma.sum(0))))
   boots.append(f1(np.einsum('bg,gij->bij',bw,mb))-f1(np.einsum('bg,gij->bij',bw,ma)))
  return dict(seed_deltas=ds,mean_delta=float(np.mean(ds)),patient_cluster95=np.percentile(np.mean(boots,axis=0),[2.5,97.5]).tolist())
 comparisons={arm+'_vs_full_ce':compare('full_ce',arm) for arm in ALL_ARMS}
 comparisons.update({arm+'_vs_grade_ce':compare('grade_ce',arm) for arm in NEW_ARMS})
 summary=[]
 for arm in ALL_ARMS:
  rr=[r for r in results if r['variant']==arm];cms=[matrices[arm,seed] for seed in cfg['seeds']]
  patient_accuracy=[float(np.mean(np.trace(m,axis1=-2,axis2=-1)/m.sum((1,2)))) for m in cms]
  malignant_recall=[float(m.sum(0)[1,1]/m.sum(0)[1].sum()) for m in cms]
  summary.append(dict(variant=arm,mean_validation_f1=float(np.mean([r['macro_f1'] for r in rr])),
      seed_sd=float(np.std([r['macro_f1'] for r in rr],ddof=1)),mean_malignant_f1=float(np.mean([r['class_f1'][1] for r in rr])),
      mean_malignant_recall=float(np.mean(malignant_recall)),mean_patient_macro_accuracy=float(np.mean(patient_accuracy)),
      **comparisons[arm+'_vs_full_ce']))
 lines=['# CLE round 6: constrain pretrained adaptation','',
        '12 new trainings and 6 reused baseline models. Same plain ResNet34/224. New arms use grade weights; no test or relabeling.',
        'Development-selected exploratory intervals condition on fitted models and do not correct repeated selection/multiplicity.','',
        '| Arm | Mean val F1 | Malignant recall | Patient mean accuracy | Delta vs full CE | Patient interval |',
        '|---|---:|---:|---:|---:|---|']
 for r in summary:lines.append(f'| {r["variant"]} | {r["mean_validation_f1"]:.6f} | {r["mean_malignant_recall"]:.6f} | {r["mean_patient_macro_accuracy"]:.6f} | {r["mean_delta"]:+.6f} | {r["patient_cluster95"]} |')
 core.save_json(STUDY/'round6_results.json',results);core.save_json(STUDY/'round6_summary.json',summary)
 core.save_json(STUDY/'round6_comparisons.json',comparisons);(STUDY/'ROUND6_REPORT.md').write_text('\n'.join(lines))
 arch.commit('results: compare pretrained adaptation constraints against both frozen CE baselines')
 core.save_json(STUDY/'status.json',dict(phase='round6_complete_needs_analysis',round=6,total=12,completed=12,reused_baseline_models=6,updated_at=core.timestamp()))


arch.summarize=summarize
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','check','worker','run']);p.add_argument('--job');a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='check':check()
 elif a.command=='worker':worker(a.job)
 else:arch.run()
