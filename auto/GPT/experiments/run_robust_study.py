#!/usr/bin/env python3
"""Round 5: training objectives and patient contribution, no new modules."""
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
STUDY=ROOT/'experiments/robust_study'
OBJECTIVES=ROOT/'experiments/robust_objectives.py'
NEW_ARMS=['label_smooth','gce','elr','patient_balance']
ALL_ARMS=['full_ce','grade_ce']+NEW_ARMS
arch.STUDY=STUDY;arch.MODEL=OBJECTIVES;arch.__file__=__file__
core.STUDY=STUDY;core.WEIGHTS=ROOT/'weights/robust_study'
ORIGINAL_VERIFY=arch.verify;ORIGINAL_DONE=arch.done;ORIGINAL_SAVE=core.save_json


def save_json(path,value):
 if Path(path)==STUDY/'status.json' and value.get('phase')=='round1_training':
  value=dict(value,phase='round5_training',round=5,reused_baseline_models=6)
 ORIGINAL_SAVE(path,value)


core.save_json=save_json


def sources():
 return dict(driver=core.digest(__file__),objectives=core.digest(OBJECTIVES),
             protocol=core.digest(ROOT/'experiments/ROUND5_PROTOCOL.md'),
             trainer=core.digest(ROOT/'experiments/run_data_study.py'),
             supervisor=core.digest(ROOT/'experiments/run_architecture_study.py'),
             backbone=core.digest(ROOT/'cle_fair_common.py'))


def verify():
 cfg=ORIGINAL_VERIFY()
 for path,sha in cfg['baseline_file_hashes'].items():assert core.digest(STUDY/path)==sha
 return cfg


arch.sources=sources;arch.verify=verify


def patient_weights(records,ids):
 counts=collections.Counter((records[i]['group_id'],records[i]['label']) for i in ids)
 totals=collections.Counter(records[i]['label'] for i in ids)
 groups=collections.Counter(c for _,c in counts)
 return {str(i):totals[records[i]['label']]/(groups[records[i]['label']]*counts[records[i]['group_id'],records[i]['label']]) for i in ids}


def baseline_files(cfg):
 import numpy as np
 import torch
 original=core.read_json(DATA/'final_jobs.json');hashes={}
 for arm,recipe in [('full_ce','full'),('grade_ce','grade_weighted')]:
  for seed in cfg['seeds']:
   job=next(j for j in original if j['arm']==recipe and j['seed']==seed)
   comp=core.read_json(DATA/'runs'/job['id']/'complete.json')
   sig=hashlib.sha256((json.dumps(job,sort_keys=True)+core.digest(DATA/'config.json')+core.digest(core.__file__)).encode()).hexdigest()
   assert comp['signature']==sig and comp['epochs_completed']==60
   ck=ROOT/comp['checkpoint'];assert core.digest(ck)==comp['checkpoint_sha256']
   out=STUDY/'baselines'/f'{arm}_s{seed}';out.mkdir(parents=True,exist_ok=True)
   cached=ROOT/'experiments/pooling_study/baselines'/f'grade_weighted_s{seed}'/'validation_predictions.csv'
   if recipe=='grade_weighted' and cached.exists():
    shutil.copyfile(cached,out/'validation_predictions.csv')
    rr=arch.read_csv(out/'validation_predictions.csv')
    p=np.array([[float(r[f'p{k}']) for k in range(3)] for r in rr]);y=np.array([int(r['label']) for r in rr])
   else:
    m=arch.ORIGINAL_MODEL(seed)
    m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=True)
    p,y,idx=core.infer(m,core.loader(cfg['val_indices'],False,seed=seed))
    core.save_csv(out/'validation_predictions.csv',core.prediction_rows(p,y,idx))
    del m;torch.cuda.empty_cache()
   result=core.metrics(p,y);assert abs(result['macro_f1']-comp['best_select_f1'])<1e-10
   core.save_json(out/'validation_result.json',dict(variant=arm,seed=seed,n=len(y),reused=True,
      source_completion=str((DATA/'runs'/job['id']/'complete.json').relative_to(ROOT)),source_signature=sig,
      checkpoint=comp['checkpoint'],checkpoint_sha256=comp['checkpoint_sha256'],**result))
   for name in ['validation_predictions.csv','validation_result.json']:
    hashes[str((out/name).relative_to(STUDY))]=core.digest(out/name)
 return hashes


def prepare():
 import numpy as np
 STUDY.mkdir(parents=True,exist_ok=True)
 if (STUDY/'config.json').exists():verify();return
 cfg=core.read_json(DATA/'config.json');records=core.read_json(DATA/'manifest.json')
 (STUDY/'manifest.json').write_bytes((DATA/'manifest.json').read_bytes())
 original=core.read_json(DATA/'final_jobs.json');ids=next(j for j in original if j['arm']=='full')['train']
 weights=patient_weights(records,ids);by_class={}
 for c in range(3):
  selected=[i for i in ids if records[i]['label']==c]
  values=np.array([weights[str(i)] for i in selected])
  assert abs(values.mean()-1)<1e-12
  group_sum=collections.defaultdict(float)
  for i in selected:group_sum[records[i]['group_id']]+=weights[str(i)]
  assert np.ptp(list(group_sum.values()))<1e-9
  by_class[c]=dict(n=len(selected),patient_groups=len(group_sum),minimum=float(values.min()),maximum=float(values.max()),
                  effective_sample_size=float(values.sum()**2/(values**2).sum()))
 cfg.update(round=5,created_at=core.timestamp(),architecture_sources=sources(),
     grade_manifest_sha256=core.digest(DATA/'train_grades.csv'),data_protocol_sha256=core.digest(DATA/'config.json'),
     variants=NEW_ARMS,comparison_arms=ALL_ARMS,new_trainings=12,reused_models=6,
     objective_hyperparameters=dict(label_smoothing=.1,gce_q=.7,elr_beta=.7,elr_lambda=3.),
     patient_weight_diagnostics=by_class,
     primary_hypothesis='Noise-adaptive objectives or balancing patient contribution improve beyond plain full-data CE, grade weighting, and generic label smoothing.',
     test_policy='No test evaluation or filtering; no labels rewritten or replaced.',
     inference='All six arms use the same plain ResNet34 at 224. Training memories are not inference inputs.',
     provenance=dict(elr_commit='934af53434a336b6db80d05d7649d23216e8ca6d',elr_loss_blob='752a81cd326ceb930b9e290c4ebd558d8dcb765f',
                     elr_config_blob='579f5ecafe15a085ad3fe85ca909a839cb82a749',license='MIT; see experiments/ELR_LICENSE.txt'))
 cfg['baseline_file_hashes']=baseline_files(cfg)
 jobs=[]
 for variant in NEW_ARMS:
  for seed in cfg['seeds']:
   j=dict(next(j for j in original if j['arm']=='full' and j['seed']==seed))
   j.update(id=f'r5_{variant}_s{seed}',phase='training_objective',arm=variant,variant=variant,recipe='full',
            score=cfg['val_indices'],architecture_sources=cfg['architecture_sources'],
            sample_weights=weights if variant=='patient_balance' else {})
   jobs.append(j)
 core.save_json(STUDY/'config.json',cfg);core.save_json(STUDY/'final_jobs.json',jobs)
 core.save_json(STUDY/'status.json',dict(phase='prepared',round=5,total=12,reused_baseline_models=6,updated_at=core.timestamp()))
 arch.commit('research: freeze robust objectives, patient balancing and six verified reused baselines')
 print(json.dumps(dict(new_jobs=12,reused_models=6,patient_weights=by_class)),flush=True)


def train_job(job):
 import torch
 import torch.nn.functional as F
 from sklearn.metrics import f1_score
 from robust_objectives import TrainingObjective
 cfg=verify();assert core.digest(core.PRETRAINED)==cfg['pretrained_sha256']
 model=arch.ORIGINAL_MODEL(job['seed'])
 criterion=TrainingObjective(job['variant']).cuda()
 out=STUDY/'runs'/job['id'];out.mkdir(parents=True,exist_ok=True)
 ckdir=core.WEIGHTS/job['id'];ckdir.mkdir(parents=True,exist_ok=True);ck=ckdir/'model_best.pth'
 weights={int(k):v for k,v in job['sample_weights'].items()}
 train=core.loader(job['train'],True,weights,job['seed']);val=core.loader(job['select'],False,seed=job['seed'])
 opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.05)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=job['epochs'])
 scaler=torch.amp.GradScaler('cuda');best=-1.;best_epoch=-1;history=[];started=time.time()
 meta=dict(job_id=job['id'],signature=core.signature(job),seed=job['seed'],arm=job['arm'],
           train_count=len(job['train']),select_count=len(job['select']),created_at=core.timestamp(),
           git_commit=subprocess.check_output([core.GIT,'-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
           pretrained_sha256=cfg['pretrained_sha256'],objective_hyperparameters=cfg['objective_hyperparameters'])
 core.save_json(out/'run_meta.json',meta)
 for epoch in range(job['epochs']):
  model.train();total_loss=total_ce=correct=n=0
  for x,y,w,idx in train:
   x,y,w=x.cuda(non_blocking=True),y.cuda(non_blocking=True),w.cuda(non_blocking=True).float()
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):
    logits=model(x)
    loss=(criterion(logits,y,idx.cuda(non_blocking=True))*w).mean()
   if not torch.isfinite(loss):raise RuntimeError('nonfinite training objective')
   scaler.scale(loss).backward();scaler.step(opt);scaler.update()
   total_loss+=float(loss.detach())*len(y)
   total_ce+=float(F.cross_entropy(logits.detach().float(),y))*len(y)
   correct+=int((logits.argmax(1)==y).sum());n+=len(y)
  p,y,_=core.infer(model,val)
  val_f1=float(f1_score(y,p.argmax(1),average='macro',labels=[0,1,2],zero_division=0));scheduler.step()
  if val_f1>best:
   best,best_epoch=val_f1,epoch;torch.save(model.state_dict(),ck.with_suffix('.tmp'));os.replace(ck.with_suffix('.tmp'),ck)
  history.append(dict(epoch=epoch,train_loss=total_loss/n,train_ce=total_ce/n,train_acc=correct/n,
                      val_macro_f1=val_f1,lr=opt.param_groups[0]['lr'],elapsed_seconds=round(time.time()-started,2)))
  core.save_csv(out/'train_log.csv',history)
  print(f'[{job["id"]}] epoch={epoch+1}/60 train_acc={correct/n:.4f} val_f1={val_f1:.4f} best={best:.4f}',flush=True)
 if job['variant']=='elr':
  assert criterion.visits.tolist()==[60]*2386
  auxiliary=ckdir/'elr_last_state.pth';torch.save(criterion.state_dict(),auxiliary)
  meta.update(elr_state=str(auxiliary.relative_to(ROOT)),elr_state_sha256=core.digest(auxiliary),elr_memory_epoch=59)
  core.save_csv(out/'elr_memory.csv',[dict(index=i,visits=int(v),t0=float(t[0]),t1=float(t[1]),t2=float(t[2]))
                                   for i,(v,t) in enumerate(zip(criterion.visits.cpu(),criterion.target.cpu()))])
 meta.update(completed_at=core.timestamp(),seconds=round(time.time()-started,2),best_select_f1=best,best_epoch=best_epoch,
             checkpoint=str(ck.relative_to(ROOT)),checkpoint_sha256=core.digest(ck),epochs_completed=len(history))
 core.save_json(out/'complete.json',meta)


def worker(job_id):
 import torch
 verify();job=next(j for j in core.read_json(STUDY/'final_jobs.json') if j['id']==job_id)
 # A valid completed training is reused when only prediction postprocessing failed.
 if not core.complete(job):train_job(job)
 out=STUDY/'runs'/job_id;comp=core.read_json(out/'complete.json')
 model=arch.ORIGINAL_MODEL(job['seed'])
 model.load_state_dict(torch.load(ROOT/comp['checkpoint'],map_location='cpu',weights_only=True),strict=True)
 p,y,idx=core.infer(model,core.loader(job['score'],False,seed=job['seed']))
 result=core.metrics(p,y)
 assert abs(result['macro_f1']-comp['best_select_f1'])<1e-10
 core.save_csv(out/'validation_predictions.csv',core.prediction_rows(p,y,idx))
 core.save_json(out/'validation_complete.json',dict(job_id=job_id,signature=core.signature(job),variant=job['variant'],
      recipe='full',seed=job['seed'],n=len(y),checkpoint_sha256=comp['checkpoint_sha256'],completed_at=core.timestamp(),
      parameters=sum(p.numel() for p in model.parameters()),modifier_parameters=0,
      diagnostics=dict(training_only=True,unchanged_inference_model=True),**result))


def done(job):
 if not ORIGINAL_DONE(job):return False
 if job['variant']=='elr':
  c=core.read_json(STUDY/'runs'/job['id']/'complete.json')
  assert core.digest(ROOT/c['elr_state'])==c['elr_state_sha256']
 return True


arch.done=done;arch.baseline_predictions=lambda:None


def check():
 import numpy as np
 import torch
 import torch.nn.functional as F
 from robust_objectives import TrainingObjective
 cfg=verify();core.set_seed(42)
 z=torch.randn(8,3,device='cuda',requires_grad=True);y=torch.arange(8,device='cuda')%3
 gce=TrainingObjective('gce').cuda();loss=gce(z,y)
 p=z.detach().softmax(1);expected=p.clone();expected[torch.arange(8),y]-=1
 expected*=p.gather(1,y[:,None]).pow(.7)
 assert torch.allclose(torch.autograd.grad(loss.sum(),z)[0],expected,atol=1e-6)
 tiny=TrainingObjective('gce',q=1e-6).cuda()
 assert torch.allclose(tiny(z,y),F.cross_entropy(z,y,reduction='none'),atol=1e-5)
 elr=TrainingObjective('elr',num_images=12).cuda();target=torch.zeros(12,3,device='cuda');ids=torch.arange(8,device='cuda')
 for _ in range(2):
  pred=z.softmax(1).clamp(1e-4,1-1e-4)
  target[ids]=.7*target[ids]+.3*pred.detach()/pred.detach().sum(1,keepdim=True)
  reference=F.cross_entropy(z,y,reduction='none')+3*(1-(target[ids]*pred).sum(1)).log()
  assert torch.allclose(elr(z,y,ids),reference,atol=1e-6)
 assert torch.equal(elr.target[8:],torch.zeros_like(elr.target[8:])) and not elr.target.requires_grad
 job=core.read_json(STUDY/'final_jobs.json')[0]
 batch,labels,_,idx=next(iter(core.loader(job['train'][:32],False,seed=42)))
 batch=batch.cuda();labels=labels.cuda();idx=idx.cuda()
 # Reused baselines require the new kernel's CE/AdamW/AMP step to be identical.
 states=[]
 for use_factory in [False,True]:
  m=arch.ORIGINAL_MODEL(42).train();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.05);s=torch.amp.GradScaler('cuda')
  with torch.autocast('cuda',dtype=torch.float16):
   logits=m(batch)
   loss=(TrainingObjective('ce')(logits,labels) if use_factory else F.cross_entropy(logits,labels,reduction='none')).mean()
  s.scale(loss).backward();s.step(opt);s.update()
  states.append({k:v.cpu().clone() for k,v in m.state_dict().items()});del m,opt,s
 assert all(torch.equal(v,states[1][k]) for k,v in states[0].items())
 del states
 records=core.read_json(STUDY/'manifest.json');pw=patient_weights(records,job['train'])
 checks=[]
 for kind in NEW_ARMS:
  m=arch.ORIGINAL_MODEL(42).train();objective=TrainingObjective(kind).cuda()
  w=torch.tensor([pw[str(i)] for i in idx.tolist()],device='cuda') if kind=='patient_balance' else torch.ones(len(idx),device='cuda')
  with torch.autocast('cuda',dtype=torch.float16):l=(objective(m(batch),labels,idx)*w).mean()
  l.backward();assert torch.isfinite(l) and torch.isfinite(m.conv1.weight.grad).all() and m.conv1.weight.grad.abs().sum()>0
  checks.append(kind);del m,objective
 for c in range(3):assert abs(np.mean([pw[str(i)] for i in job['train'] if records[i]['label']==c])-1)<1e-12
 core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),objectives=checks,
    checks=['GCE analytical gradient and CE limit','ELR official-equation values and index-isolated detached EMA',
            'CE AdamW/AMP update bitwise identical to immutable trainer step','real-image finite objective/backward',
            'patient class mass and within-class patient contribution balanced','six reused baseline checkpoint hashes and validation reproduction']))
 arch.commit('test: validate robust-objective math, EMA indexing and baseline training-step equivalence')
 print('PASS robust training preflight',flush=True)


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
 comparisons.update({arm+'_vs_label_smooth':compare('label_smooth',arm) for arm in ['gce','elr','patient_balance']})
 summary=[]
 for arm in ALL_ARMS:
  rr=[r for r in results if r['variant']==arm];cms=[matrices[arm,seed] for seed in cfg['seeds']]
  patient_accuracy=[float(np.mean(np.trace(m,axis1=-2,axis2=-1)/m.sum((1,2)))) for m in cms]
  malignant_recall=[float(m.sum(0)[1,1]/m.sum(0)[1].sum()) for m in cms]
  summary.append(dict(variant=arm,mean_validation_f1=float(np.mean([r['macro_f1'] for r in rr])),
      seed_sd=float(np.std([r['macro_f1'] for r in rr],ddof=1)),mean_malignant_f1=float(np.mean([r['class_f1'][1] for r in rr])),
      mean_malignant_recall=float(np.mean(malignant_recall)),mean_patient_macro_accuracy=float(np.mean(patient_accuracy)),
      **comparisons[arm+'_vs_full_ce']))
 lines=['# CLE round 5: noise-adaptive training and patient balance','',
        '12 new trainings and 6 reused baseline models. Same plain ResNet34/224; no test evaluation or labels rewritten.',
        'Development-selected exploratory intervals condition on fitted models and do not correct repeated selection/multiplicity.','',
        '| Arm | Mean val F1 | Malignant recall | Patient mean accuracy | Delta vs full CE | Patient interval |',
        '|---|---:|---:|---:|---:|---|']
 for r in summary:lines.append(f'| {r["variant"]} | {r["mean_validation_f1"]:.6f} | {r["mean_malignant_recall"]:.6f} | {r["mean_patient_macro_accuracy"]:.6f} | {r["mean_delta"]:+.6f} | {r["patient_cluster95"]} |')
 core.save_json(STUDY/'round5_results.json',results);core.save_json(STUDY/'round5_summary.json',summary)
 core.save_json(STUDY/'round5_comparisons.json',comparisons);(STUDY/'ROUND5_REPORT.md').write_text('\n'.join(lines))
 arch.commit('results: compare robust objectives and patient contribution against both frozen CE baselines')
 core.save_json(STUDY/'status.json',dict(phase='round5_complete_needs_analysis',round=5,total=12,completed=12,reused_baseline_models=6,updated_at=core.timestamp()))


arch.summarize=summarize
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','check','worker','run']);p.add_argument('--job');a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='check':check()
 elif a.command=='worker':worker(a.job)
 else:arch.run()
