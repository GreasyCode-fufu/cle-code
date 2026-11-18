#!/usr/bin/env python3
"""Round 2: independently frozen pooling study; no test evaluation."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import run_architecture_study as arch
core=arch.core
DATA=ROOT/'experiments/data_study'
PREVIOUS=ROOT/'experiments/architecture_study'
STUDY=ROOT/'experiments/pooling_study'
MODEL=ROOT/'models/optimize/cle_evidence_pool.py'
arch.STUDY=STUDY;core.STUDY=STUDY;core.WEIGHTS=ROOT/'weights/pooling_study'
arch.MODEL=MODEL
# The inherited supervisor launches this round's driver, not the first round.
arch.__file__=__file__
ORIGINAL_VERIFY=arch.verify
ORIGINAL_WORKER=arch.worker
VARIANTS=['gem_control','attention_control','cle_evidence','cle_no_frequency','cle_fixed_blend','cle_no_context']
ORIGINAL_SAVE_JSON=core.save_json


def save_json(path,value):
 if Path(path)==STUDY/'status.json' and value.get('phase')=='round1_training':
  value=dict(value,phase='round2_training',round=2)
 ORIGINAL_SAVE_JSON(path,value)


core.save_json=save_json


def sources():
 return dict(pooling_driver=core.digest(__file__),pooling_model=core.digest(MODEL),
             architecture_driver=core.digest(ROOT/'experiments/run_architecture_study.py'),
             trainer=core.digest(ROOT/'experiments/run_data_study.py'))


def verify():
 cfg=ORIGINAL_VERIFY()
 for rel,sha in cfg['baseline_file_hashes'].items():assert core.digest(STUDY/rel)==sha
 return cfg


def model_for(seed,variant,position=3,control_width=64):
 from models.optimize.cle_evidence_pool import EvidenceResNet
 return EvidenceResNet(arch.ORIGINAL_MODEL(seed),variant,control_width).cuda()


arch.sources=sources;arch.verify=verify;arch.model_for=model_for


def prepare():
 from models.optimize.cle_evidence_pool import EvidencePooling
 STUDY.mkdir(parents=True,exist_ok=True)
 if (STUDY/'config.json').exists():verify();return
 cfg=core.read_json(DATA/'config.json')
 target=sum(p.numel() for p in EvidencePooling('cle_evidence').parameters())
 options={w:sum(p.numel() for p in EvidencePooling('attention_control',control_width=w).parameters()) for w in range(8,257,8)}
 width=min(options,key=lambda w:abs(options[w]-target))
 cfg.update(round=2,created_at=core.timestamp(),architecture_sources=sources(),
            grade_manifest_sha256=core.digest(DATA/'train_grades.csv'),data_protocol_sha256=core.digest(DATA/'config.json'),
            variants=VARIANTS,recipes=['grade_weighted'],control_width=width,
            full_extra_parameters=target,control_extra_parameters=options[width],
            parameter_match_relative_error=abs(options[width]-target)/target,
            primary_hypothesis='Context-constrained spatial aggregation with morphology-band cues improves beyond average pooling and matched gated attention.',
            diagnostic='Fixed-backbone extreme/GeM pooling generally degraded development accuracy; local responses still existed in most errors. This motivates bounded learned aggregation, not unbounded top-k.',
            recipe_reason='Weighted recipe chosen from prior development means (0.8820 versus A-C 0.8724), not test ranking.',
            test_policy='No test forward pass in this driver; same 538 development images for all candidates.',
            selection_policy='Three paired seeds, complete development set, matched attention control, three prespecified mechanism ablations. Exploratory intervals, independent confirmation required.')
 (STUDY/'manifest.json').write_bytes((DATA/'manifest.json').read_bytes())
 original=core.read_json(DATA/'final_jobs.json');jobs=[]
 for variant in VARIANTS:
  for seed in [42,43,44]:
   old=next(j for j in original if j['arm']=='grade_weighted' and j['seed']==seed)
   j=dict(old)
   j.update(id=f'r2_grade_weighted_{variant}_s{seed}',phase='architecture',arm='grade_weighted/'+variant,
            recipe='grade_weighted',variant=variant,position=4,control_width=width,
            score=cfg['val_indices'],architecture_sources=cfg['architecture_sources'])
   jobs.append(j)
 hashes={}
 for seed in [42,43,44]:
  rel=Path('baselines')/f'grade_weighted_s{seed}'
  (STUDY/rel).mkdir(parents=True,exist_ok=True)
  for name in ['validation_predictions.csv','validation_result.json']:
   shutil.copyfile(PREVIOUS/rel/name,STUDY/rel/name)
   hashes[str(rel/name)]=core.digest(STUDY/rel/name)
 cfg['baseline_file_hashes']=hashes
 core.save_json(STUDY/'config.json',cfg);core.save_json(STUDY/'final_jobs.json',jobs)
 core.save_json(STUDY/'status.json',dict(phase='prepared',round=2,total=18,updated_at=core.timestamp()))
 arch.commit('research: freeze second-round bounded evidence pooling and reuse validated development baselines')
 print(json.dumps(dict(jobs=18,attention_width=width,control_parameters=options[width],full_parameters=target)),flush=True)


def worker(job):
 ORIGINAL_WORKER(job)
 p=STUDY/'runs'/job/'validation_complete.json'
 r=core.read_json(p);d=r['diagnostics']
 d['global_pool_share_mean']=d.pop('route_local_mean')
 d['pooled_shift_proxy_mean']=d.pop('stream_perturbation_mean')
 d['pooled_shift_proxy_quantiles']=d.pop('stream_perturbation_quantiles')
 d['interpretation']='Broadcast descriptor shift preserves pooling equivalence. Not physical image noise or a diagnosis-quality score.'
 core.save_json(p,r)


def check():
 import torch
 import torch.nn.functional as F
 from models.optimize.cle_evidence_pool import haar_bands,EvidencePooling
 cfg=verify();core.set_seed(42);checked=[]
 for hw in [7,14]:
  x=torch.randn(2,8,hw,hw,dtype=torch.float64,requires_grad=True)
  bands=haar_bands(x)
  assert torch.allclose(x.square().sum(),bands.square().sum(),atol=1e-9), 'Haar energy not preserved'
  bands.square().mean().backward();assert torch.isfinite(x.grad).all()
 for variant in VARIANTS:
  module=EvidencePooling(variant,control_width=cfg['control_width']).cuda()
  x4=torch.rand(4,512,7,7,device='cuda',requires_grad=True)
  x3=torch.rand(4,256,14,14,device='cuda',requires_grad=True)
  opt=torch.optim.AdamW(module.parameters(),lr=1e-4)
  for _ in range(3):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):delta=module(x4,x3)
   assert delta.shape==(4,512,1,1) and torch.isfinite(delta).all()
   loss=(x4.mean((2,3),keepdim=True)+delta).square().mean();loss.backward();opt.step()
  assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())
  if module.last_attention is not None:
   assert torch.allclose(module.last_attention.sum((2,3)),torch.ones(4,1,device='cuda'),atol=1e-5)
   assert module.last_routing[:,0].min()>=.5 and module.last_routing[:,0].max()<=1
  if variant=='cle_evidence':
   assert module.frequency_coefficient.grad is not None and module.frequency_coefficient.grad.abs()>0
  checked.append(variant)
  del module,opt
 base=arch.ORIGINAL_MODEL(42).eval()
 base_state={k:v.cpu().clone() for k,v in base.state_dict().items()}
 batch=next(iter(core.loader(cfg['val_indices'][:32],False,seed=42)))[0].cuda()
 for variant in VARIANTS:
  model=model_for(42,variant,control_width=cfg['control_width']).eval()
  assert all(torch.equal(v.cpu(),base_state[k]) for k,v in model.backbone.state_dict().items())
  model.modifier.enabled=False
  with torch.inference_mode():assert torch.equal(model(batch),model.backbone(batch))
  model.modifier.enabled=True
  with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model.train()(batch),torch.zeros(32,device='cuda',dtype=torch.long))
  loss.backward();assert torch.isfinite(loss)
  del model
 core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),variants=checked,
   checks=['orthonormal Haar energy preservation and gradients including odd shape','finite AMP pooling updates',
           'normalized attention and at least half GAP share','nonzero frequency coefficient gradient',
           'identical pretrained backbone/head per seed','disabled pooling equals original backbone','full-image CUDA backward']))
 arch.commit('test: validate bounded evidence pooling, wavelet basis and paired initialization')
 print('PASS pooling preflight',flush=True)


def summarize():
 import numpy as np
 cfg=verify();jobs=core.read_json(STUDY/'final_jobs.json')
 records=[core.read_json(STUDY/'runs'/j['id']/'validation_complete.json') for j in jobs]
 read=arch.read_csv;ids=None;groups=None;bw=None
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
 matrices={}
 for seed in [42,43,44]:
  matrices['gap',seed]=matrix(read(STUDY/'baselines'/f'grade_weighted_s{seed}'/'validation_predictions.csv'))
  for v in VARIANTS:
   j=next(j for j in jobs if j['variant']==v and j['seed']==seed)
   matrices[v,seed]=matrix(read(STUDY/'runs'/j['id']/'validation_predictions.csv'))
 def compare(a,b):
  ds=[];boots=[]
  for seed in [42,43,44]:
   ma,mb=matrices[a,seed],matrices[b,seed]
   ds.append(float(f1(mb.sum(0))-f1(ma.sum(0))))
   boots.append(f1(np.einsum('bg,gij->bij',bw,mb))-f1(np.einsum('bg,gij->bij',bw,ma)))
  return dict(seed_deltas=ds,mean_delta=float(np.mean(ds)),patient_cluster95=np.percentile(np.mean(boots,axis=0),[2.5,97.5]).tolist())
 comparisons={v+'_vs_gap':compare('gap',v) for v in VARIANTS}
 comparisons['cle_vs_attention']=compare('attention_control','cle_evidence')
 comparisons['cle_vs_gem']=compare('gem_control','cle_evidence')
 for v in ['cle_no_frequency','cle_fixed_blend','cle_no_context']:comparisons['cle_vs_'+v]=compare(v,'cle_evidence')
 lines=['# CLE round 2: bounded spatial evidence pooling','',
        'Development-selected exploratory results only. Patient intervals condition on fitted pairs; selection/multiple comparisons need independent confirmation.',
        'No test images were evaluated.','',
        '| Variant | Mean val F1 | Mean malignant F1 | Delta vs GAP | Patient interval | Extra parameters |',
        '|---|---:|---:|---:|---|---:|']
 summary=[]
 for v in VARIANTS:
  rr=[r for r in records if r['variant']==v];d=comparisons[v+'_vs_gap']
  entry=dict(variant=v,mean_validation_f1=float(np.mean([r['macro_f1'] for r in rr])),
             mean_malignant_f1=float(np.mean([r['class_f1'][1] for r in rr])),modifier_parameters=rr[0]['modifier_parameters'],**d)
  summary.append(entry)
  lines.append(f'| {v} | {entry["mean_validation_f1"]:.6f} | {entry["mean_malignant_f1"]:.6f} | {d["mean_delta"]:+.6f} | {d["patient_cluster95"]} | {entry["modifier_parameters"]} |')
 core.save_json(STUDY/'round2_results.json',records);core.save_json(STUDY/'round2_summary.json',summary)
 core.save_json(STUDY/'round2_comparisons.json',comparisons)
 (STUDY/'ROUND2_REPORT.md').write_text('\n'.join(lines))
 arch.commit('results: summarize pooling mechanisms against GAP and matched attention controls')
 core.save_json(STUDY/'status.json',dict(phase='round2_complete_needs_analysis',round=2,total=18,completed=18,updated_at=core.timestamp(),
                                       report='experiments/pooling_study/ROUND2_REPORT.md'))


arch.summarize=summarize

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','check','worker','run']);p.add_argument('--job');a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='check':check()
 elif a.command=='worker':worker(a.job)
 else:arch.run()
