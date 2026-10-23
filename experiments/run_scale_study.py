#!/usr/bin/env python3
"""Frozen round 3: shared-convolution resolution paths, development only."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_architecture_study as arch
core=arch.core
DATA=ROOT/'experiments/data_study'
PREVIOUS=ROOT/'experiments/pooling_study'
STUDY=ROOT/'experiments/scale_study'
MODEL=ROOT/'models/optimize/cle_scale_route.py'
VARIANTS=['dilated_control','fixed_dual','sk_dual','cle_contrast','cle_no_dilation','static_dual']
arch.STUDY=STUDY;arch.MODEL=MODEL;arch.__file__=__file__
core.STUDY=STUDY;core.WEIGHTS=ROOT/'weights/scale_study'
ORIGINAL_VERIFY=arch.verify
ORIGINAL_SAVE_JSON=core.save_json


def save_json(path,value):
 if Path(path)==STUDY/'status.json' and value.get('phase')=='round1_training':
  value=dict(value,phase='round3_training',round=3)
 ORIGINAL_SAVE_JSON(path,value)


core.save_json=save_json


def sources():
 return dict(scale_driver=core.digest(__file__),scale_model=core.digest(MODEL),
             protocol=core.digest(ROOT/'experiments/ROUND3_PROTOCOL.md'),
             architecture_driver=core.digest(ROOT/'experiments/run_architecture_study.py'),
             trainer=core.digest(ROOT/'experiments/run_data_study.py'))


def verify():
 cfg=ORIGINAL_VERIFY()
 for rel,sha in cfg['baseline_file_hashes'].items():assert core.digest(STUDY/rel)==sha
 return cfg


def model_for(seed,variant,position=4,control_width=64):
 from models.optimize.cle_scale_route import ScaleRouteResNet
 return ScaleRouteResNet(arch.ORIGINAL_MODEL(seed),variant,control_width).cuda()


arch.sources=sources;arch.verify=verify;arch.model_for=model_for


def prepare():
 STUDY.mkdir(parents=True,exist_ok=True)
 if (STUDY/'config.json').exists():verify();return
 cfg=core.read_json(DATA/'config.json')
 cfg.update(round=3,created_at=core.timestamp(),architecture_sources=sources(),
            grade_manifest_sha256=core.digest(DATA/'train_grades.csv'),data_protocol_sha256=core.digest(DATA/'config.json'),
            variants=VARIANTS,recipes=['grade_weighted'],control_width=64,
            primary_hypothesis='Preserving 14x14 layer4 morphology plus contrast-conditioned selection improves beyond a high-resolution-only path, fixed fusion and a matched SK-style router.',
            recipe_reason='Prior full-development data study selected weighting, not test performance.',
            selection_policy='Three paired seeds, complete development patients, primary full-versus-SK/fixed/dilated contrasts; exploratory intervals need independent confirmation.',
            test_policy='No test evaluation or filtering; full 538 development images.',
            design_limits='Only layer4 stride/sampling and late descriptor routing differ; this is not a DRN or SKNet reproduction. Original input resize/crop remains 256/224.',
            normalization='Separate copied high-path BatchNorm in dual variants; convolution weights shared. Single-path control uses original BatchNorm only.',
            provenance={'library79_sha256':'d3d795e98bcdc93b33a1c0750689e2030fa265d9f010d0d037767e6dbaacbc50',
                        'library84_sha256':'935c7b8764ed383bc00e203dba89dbdf01fcb18f80578ee136b37a467e90e20d',
                        'implementation':'Independent equations; no imported/copied third-party code; library venue labels unverified.'})
 (STUDY/'manifest.json').write_bytes((DATA/'manifest.json').read_bytes())
 original=core.read_json(DATA/'final_jobs.json');jobs=[];hashes={}
 for variant in VARIANTS:
  for seed in cfg['seeds']:
   j=dict(next(j for j in original if j['arm']=='grade_weighted' and j['seed']==seed))
   j.update(id=f'r3_grade_weighted_{variant}_s{seed}',phase='architecture',arm='grade_weighted/'+variant,
            recipe='grade_weighted',variant=variant,position=4,control_width=64,
            score=cfg['val_indices'],architecture_sources=cfg['architecture_sources'])
   jobs.append(j)
 for seed in cfg['seeds']:
  rel=Path('baselines')/f'grade_weighted_s{seed}';(STUDY/rel).mkdir(parents=True,exist_ok=True)
  for name in ['validation_predictions.csv','validation_result.json']:
   shutil.copyfile(PREVIOUS/rel/name,STUDY/rel/name);hashes[str(rel/name)]=core.digest(STUDY/rel/name)
 cfg['baseline_file_hashes']=hashes
 core.save_json(STUDY/'config.json',cfg);core.save_json(STUDY/'final_jobs.json',jobs)
 core.save_json(STUDY/'status.json',dict(phase='prepared',round=3,total=18,updated_at=core.timestamp()))
 arch.commit('research: freeze shared-weight scale routing controls and 18 development-only jobs')
 print('Prepared 18 scale-study jobs',flush=True)


def worker(job_id):
 import numpy as np
 import torch
 verify()
 j=next(j for j in core.read_json(STUDY/'final_jobs.json') if j['id']==job_id)
 core.new_model=lambda seed:model_for(seed,j['variant'],control_width=j['control_width'])
 core.run_worker(job_id)
 out=STUDY/'runs'/job_id;completion=core.read_json(out/'complete.json')
 model=core.new_model(j['seed'])
 model.load_state_dict(torch.load(ROOT/completion['checkpoint'],map_location='cpu',weights_only=True),strict=True)
 model.modifier.capture=True
 collected=[]
 def hook(module,inputs,output):collected.append(dict(module.last_stats))
 handle=model.modifier.register_forward_hook(hook)
 p,y,idx=core.infer(model,core.loader(j['score'],False,seed=j['seed']))
 handle.remove();model.modifier.capture=False
 core.save_csv(out/'validation_predictions.csv',core.prediction_rows(p,y,idx))
 diag=dict(high_shape=collected[0]['high_shape'],
           note='Off uses the same fitted shared convolutions; not a retrained ablation. No off score for single-path dilation.')
 for key in ['high_share','high_share_channel_sd','descriptor_shift']:
  v=[v for batch in collected for v in batch.get(key,[])]
  if v:diag[key]=dict(mean=float(np.mean(v)),quantiles=np.percentile(v,[0,25,50,75,100]).tolist())
 if j['variant']!='dilated_control':
  diag['low_shape']=collected[0]['low_shape']
  model.modifier.enabled=False
  po,yo,idxo=core.infer(model,core.loader(j['score'],False,seed=j['seed']))
  assert idx==idxo and np.array_equal(y,yo)
  diag['off_same_checkpoint_metrics']=core.metrics(po,yo)
  diag['off_prediction_disagreement']=float(np.mean(p.argmax(1)!=po.argmax(1)))
 core.save_json(out/'validation_complete.json',dict(signature=core.signature(j),job_id=job_id,
     recipe=j['recipe'],variant=j['variant'],seed=j['seed'],n=len(y),checkpoint_sha256=completion['checkpoint_sha256'],
     completed_at=core.timestamp(),parameters=sum(v.numel() for v in model.parameters()),
     modifier_parameters=sum(v.numel() for v in model.modifier.parameters()),diagnostics=diag,**core.metrics(p,y)))


def check():
 import numpy as np
 import torch
 import torch.nn.functional as F
 from torch.utils.flop_counter import FlopCounterMode
 from models.optimize.cle_scale_route import HighResolutionPath
 cfg=verify();core.set_seed(42)
 base=arch.ORIGINAL_MODEL(42).eval();stage=base.layer4
 # Independent nn.Conv2d reference, not a second copy of functional forward.
 ref=copy.deepcopy(stage).eval()
 for i,b in enumerate(ref):
  b.conv1.stride=(1,1);b.conv1.dilation=(1,1) if i==0 else (2,2);b.conv1.padding=b.conv1.dilation
  b.conv2.dilation=(2,2);b.conv2.padding=(2,2)
  if b.downsample is not None:b.downsample[0].stride=(1,1)
 hp=HighResolutionPath(stage,True).cuda().eval()
 x=torch.randn(2,256,14,14,device='cuda',requires_grad=True);xr=x.detach().clone().requires_grad_()
 a=hp(x,stage);b=ref(xr)
 assert torch.allclose(a,b,atol=2e-5,rtol=2e-5)
 ga=torch.autograd.grad(a.square().mean(),(x,stage[0].conv1.weight))
 gb=torch.autograd.grad(b.square().mean(),(xr,ref[0].conv1.weight))
 assert all(torch.allclose(u,v,atol=2e-5,rtol=2e-4) for u,v in zip(ga,gb))
 base_state={k:v.cpu().clone() for k,v in base.state_dict().items()}
 del hp,ref,base,stage,a,b,x,xr,ga,gb
 job=core.read_json(STUDY/'final_jobs.json')[0]
 batch,labels,_,_=next(iter(core.loader(job['train'][:32],False,seed=42)))
 batch=batch.cuda();labels=labels.cuda();budget=[]
 for variant in VARIANTS:
  model=model_for(42,variant).eval()
  assert all(torch.equal(v.cpu(),base_state[k]) for k,v in model.backbone.state_dict().items())
  model.modifier.enabled=False
  with torch.inference_mode():assert torch.equal(model(batch),model.backbone(batch))
  model.modifier.enabled=True
  with torch.inference_mode(),FlopCounterMode(display=False) as counter:model(batch[:1])
  budget.append(dict(variant=variant,parameters=sum(p.numel() for p in model.parameters()),
                     extra_parameters=sum(p.numel() for p in model.modifier.parameters()),
                     conv_matmul_flops=counter.get_total_flops()))
  model.modifier.capture=True
  with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model.train()(batch),labels)
  loss.backward();assert torch.isfinite(loss)
  assert model.backbone.layer4[0].conv1.weight.grad.abs().sum()>0
  if hasattr(model.modifier,'router'):assert model.modifier.router[-1].weight.grad.abs().sum()>0
  if model.modifier.high.separate_norm:assert model.modifier.high.norms[0][0].weight.grad.abs().sum()>0
  assert model.modifier.last_stats['high_shape']==[14,14]
  if variant!='dilated_control':
   r=model.modifier.last_routing
   assert torch.allclose(r.sum(1),torch.ones_like(r[:,0]),atol=1e-6)
   assert torch.allclose(r,torch.full_like(r,.5))
  del model;torch.cuda.empty_cache()
 full=next(r for r in budget if r['variant']=='cle_contrast')
 control=next(r for r in budget if r['variant']=='sk_dual')
 assert full['parameters']==control['parameters'] and abs(full['conv_matmul_flops']/control['conv_matmul_flops']-1)<.01
 core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),budget=budget,
      checks=['functional high path matches independent Conv2d reference forward and gradients',
              'paired original backbone and head','disabled module equals original backbone exactly',
              'finite real-training-image AMP backward','shared convolution and separate high BatchNorm gradients',
              'nonzero final-router gradient at initial 0.5 routing','14x14 high versus 7x7 low',
              'identical full/SK parameter count and matched counted convolution/matmul FLOPs'],
      budget_limit='Counted FLOPs omit some pooling, normalization and elementwise costs; no zero-inference-cost claim.'))
 arch.commit('test: verify high-resolution sharing, gradients, normalization and matched routing budgets')
 print(json.dumps(dict(passed=True,budget=budget)),flush=True)


def summarize():
 import numpy as np
 cfg=verify();jobs=core.read_json(STUDY/'final_jobs.json')
 records=[core.read_json(STUDY/'runs'/j['id']/'validation_complete.json') for j in jobs]
 matrices={};ids=None;groups=None;bw=None
 def matrix(rows):
  nonlocal ids,groups,bw
  rows=sorted(rows,key=lambda r:int(r['index']))
  if ids is None:
   ids=[r['index'] for r in rows];groups=sorted({r['group_id'] for r in rows})
   bw=np.random.default_rng(20261024).multinomial(len(groups),np.ones(len(groups))/len(groups),size=20000)
  assert ids==[r['index'] for r in rows]
  m=np.zeros((len(groups),3,3))
  np.add.at(m,([groups.index(r['group_id']) for r in rows],[int(r['label']) for r in rows],[int(r['pred']) for r in rows]),1)
  return m
 def f1(m):
  d=m.sum(-1)+m.sum(-2)
  return np.mean(np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),d,out=np.zeros_like(d),where=d!=0),axis=-1)
 for seed in cfg['seeds']:
  matrices['gap',seed]=matrix(arch.read_csv(STUDY/'baselines'/f'grade_weighted_s{seed}'/'validation_predictions.csv'))
  for variant in VARIANTS:
   j=next(j for j in jobs if j['variant']==variant and j['seed']==seed)
   matrices[variant,seed]=matrix(arch.read_csv(STUDY/'runs'/j['id']/'validation_predictions.csv'))
 def compare(a,b):
  ds=[];boots=[]
  for seed in cfg['seeds']:
   ma,mb=matrices[a,seed],matrices[b,seed]
   ds.append(float(f1(mb.sum(0))-f1(ma.sum(0))))
   boots.append(f1(np.einsum('bg,gij->bij',bw,mb))-f1(np.einsum('bg,gij->bij',bw,ma)))
  return dict(seed_deltas=ds,mean_delta=float(np.mean(ds)),patient_cluster95=np.percentile(np.mean(boots,axis=0),[2.5,97.5]).tolist())
 comparisons={v+'_vs_gap':compare('gap',v) for v in VARIANTS}
 for v in VARIANTS:
  if v!='cle_contrast':comparisons['cle_vs_'+v]=compare(v,'cle_contrast')
 summary=[]
 for v in VARIANTS:
  rr=[r for r in records if r['variant']==v]
  summary.append(dict(variant=v,mean_validation_f1=float(np.mean([r['macro_f1'] for r in rr])),
    seed_sd=float(np.std([r['macro_f1'] for r in rr],ddof=1)),mean_malignant_f1=float(np.mean([r['class_f1'][1] for r in rr])),
    modifier_parameters=rr[0]['modifier_parameters'],**comparisons[v+'_vs_gap']))
 lines=['# CLE round 3: shared-weight scale routing','',
        'Exploratory development-selected scores; intervals condition on fitted models, without model-selection/multiplicity correction. No test evaluation.','',
        '| Variant | Mean val F1 | Malignant F1 | Delta vs GAP | Patient interval | Extra parameters |',
        '|---|---:|---:|---:|---|---:|']
 for r in summary:lines.append(f'| {r["variant"]} | {r["mean_validation_f1"]:.6f} | {r["mean_malignant_f1"]:.6f} | {r["mean_delta"]:+.6f} | {r["patient_cluster95"]} | {r["modifier_parameters"]} |')
 core.save_json(STUDY/'round3_results.json',records);core.save_json(STUDY/'round3_summary.json',summary)
 core.save_json(STUDY/'round3_comparisons.json',comparisons)
 (STUDY/'ROUND3_REPORT.md').write_text('\n'.join(lines))
 arch.commit('results: summarize resolution and contrast-routing controls on complete development patients')
 core.save_json(STUDY/'status.json',dict(phase='round3_complete_needs_analysis',round=3,total=18,completed=18,updated_at=core.timestamp()))


arch.summarize=summarize
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','check','worker','run']);p.add_argument('--job');a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='check':check()
 elif a.command=='worker':worker(a.job)
 else:arch.run()
