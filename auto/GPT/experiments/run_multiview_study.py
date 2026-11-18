#!/usr/bin/env python3
"""Frozen round 4: original-pixel multiview frames, development only."""
import argparse
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
STUDY=ROOT/'experiments/multiview_study'
MODEL=ROOT/'models/optimize/cle_multiview.py'
NEWCACHE=ROOT/'.cache_multiview_study'
VARIANTS=['global224','single448','tile_fixed','tile_attention','tile_context','tile_no_context']
arch.STUDY=STUDY;arch.MODEL=MODEL;arch.__file__=__file__
core.STUDY=STUDY;core.WEIGHTS=ROOT/'weights/multiview_study'
ORIGINAL_VERIFY=arch.verify
ORIGINAL_SAVE_JSON=core.save_json


def save_json(path,value):
 if Path(path)==STUDY/'status.json' and value.get('phase')=='round1_training':
  value=dict(value,phase='round4_training',round=4)
 ORIGINAL_SAVE_JSON(path,value)


core.save_json=save_json


def sources():
 return dict(view_driver=core.digest(__file__),view_model=core.digest(MODEL),
             protocol=core.digest(ROOT/'experiments/ROUND4_PROTOCOL.md'),
             architecture_driver=core.digest(ROOT/'experiments/run_architecture_study.py'),
             trainer=core.digest(ROOT/'experiments/run_data_study.py'))


def verify():
 cfg=ORIGINAL_VERIFY()
 for rel,sha in cfg['baseline_file_hashes'].items():assert core.digest(STUDY/rel)==sha
 meta=core.read_json(NEWCACHE/'meta.json')
 assert core.digest(NEWCACHE/'meta.json')==cfg['view_cache_meta_sha256']
 assert meta['cache_sha256']==cfg['view_cache_sha256']
 stat=(NEWCACHE/'resized512_uint8.npy').stat()
 assert stat.st_size==meta['size_bytes'] and stat.st_mtime_ns==meta['mtime_ns']
 return cfg


def model_for(seed,variant,position=4,control_width=64):
 from models.optimize.cle_multiview import MultiViewResNet
 return MultiViewResNet(arch.ORIGINAL_MODEL(seed),variant,control_width).cuda()


arch.sources=sources;arch.verify=verify;arch.model_for=model_for


def prepare_cache():
 import numpy as np
 from PIL import Image
 NEWCACHE.mkdir(exist_ok=True)
 target=NEWCACHE/'resized512_uint8.npy';metapath=NEWCACHE/'meta.json'
 cfg=core.read_json(DATA/'config.json')
 if metapath.exists():
  meta=core.read_json(metapath)
  assert meta['manifest_sha256']==cfg['manifest_sha256'] and core.digest(target)==meta['cache_sha256']
  return meta
 records=core.read_json(DATA/'manifest.json')
 ids=[r['index'] for r in records if r['split'] in ['train','val']]
 assert ids==list(range(2924))
 tmp=NEWCACHE/'resized512_uint8.tmp.npy'
 arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(len(ids),512,512,3))
 for idx in ids:
  with Image.open(ROOT/records[idx]['path']) as im:
   arr[idx]=np.asarray(im.convert('RGB').resize((512,512),Image.Resampling.BILINEAR))
  if idx%500==0:print(f'[view cache] {idx}/{len(ids)}',flush=True)
 arr.flush();del arr;tmp.replace(target)
 stat=target.stat()
 meta=dict(manifest_sha256=cfg['manifest_sha256'],count=len(ids),excluded_split='test',
           transform='PIL RGB Resize(512,512) BILINEAR',cache_sha256=core.digest(target),
           size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
 core.save_json(metapath,meta)
 return meta


def build_dataset(indices,training=False,sample_weights=None):
 import numpy as np
 import torch
 from PIL import Image
 from torchvision import transforms as T
 records=core.read_json(STUDY/'manifest.json')
 assert all(0<=i<2924 and records[i]['split'] in ['train','val'] for i in indices)
 transform=T.Compose(([T.RandomHorizontalFlip(),T.RandomVerticalFlip(),T.RandomRotation(15),T.RandomCrop(448)]
                       if training else [T.CenterCrop(448)])+
                      [T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 class Images(torch.utils.data.Dataset):
  def __init__(self):self.array=np.load(NEWCACHE/'resized512_uint8.npy',mmap_mode='r')
  def __len__(self):return len(indices)
  def __getitem__(self,pos):
   idx=indices[pos]
   return transform(Image.fromarray(self.array[idx])),records[idx]['label'],float((sample_weights or {}).get(idx,1.)),idx
 return Images()


core.build_dataset=build_dataset


def prepare():
 meta=prepare_cache()
 STUDY.mkdir(parents=True,exist_ok=True)
 if (STUDY/'config.json').exists():verify();return
 cfg=core.read_json(DATA/'config.json')
 cfg.update(round=4,created_at=core.timestamp(),architecture_sources=sources(),
            grade_manifest_sha256=core.digest(DATA/'train_grades.csv'),data_protocol_sha256=core.digest(DATA/'config.json'),
            variants=VARIANTS,recipes=['grade_weighted'],control_width=64,img_size=448,resize=512,
            view_cache_meta_sha256=core.digest(NEWCACHE/'meta.json'),view_cache_sha256=meta['cache_sha256'],
            primary_hypothesis='Jointly trained global context and original-pixel detail tiles improve beyond matched global preprocessing, a single 448 input, fixed fusion and matched gated MIL-style attention.',
            recipe_reason='Prior full-development data study selected weighting, not test performance.',
            selection_policy='Three paired seeds, complete development patients; full-versus-global224/single448/fixed/attention. Exploratory intervals need independent confirmation.',
            test_policy='No test evaluation or filtering; full 538 development images.',
            design_limits='All variants share 512 resize/448 crop; global control downsamples the same view to 224. Legacy 256/224 baseline is contextual only. New view sampling can change effective scale; larger input is not presumed better.',
            normalization='Shared CNN convolution/head; copied local-domain BatchNorm for tiled models. No per-tile labels or manual quality labels.',
            provenance={'library60a_sha256':'87257c1d2d73a2a0527e74c56770ee59cd76e042c8e3608914126e861a723537',
                        'implementation':'Independent multiview equations; no imported/copied CSA/TransMIL code, not a faithful reproduction.'})
 (STUDY/'manifest.json').write_bytes((DATA/'manifest.json').read_bytes())
 original=core.read_json(DATA/'final_jobs.json');jobs=[];hashes={}
 for variant in VARIANTS:
  for seed in cfg['seeds']:
   j=dict(next(j for j in original if j['arm']=='grade_weighted' and j['seed']==seed))
   j.update(id=f'r4_grade_weighted_{variant}_s{seed}',phase='architecture',arm='grade_weighted/'+variant,
            recipe='grade_weighted',variant=variant,position=4,control_width=64,
            score=cfg['val_indices'],architecture_sources=cfg['architecture_sources'])
   jobs.append(j)
 for seed in cfg['seeds']:
  rel=Path('baselines')/f'grade_weighted_s{seed}';(STUDY/rel).mkdir(parents=True,exist_ok=True)
  for name in ['validation_predictions.csv','validation_result.json']:
   shutil.copyfile(PREVIOUS/rel/name,STUDY/rel/name);hashes[str(rel/name)]=core.digest(STUDY/rel/name)
 cfg['baseline_file_hashes']=hashes
 core.save_json(STUDY/'config.json',cfg);core.save_json(STUDY/'final_jobs.json',jobs)
 core.save_json(STUDY/'status.json',dict(phase='prepared',round=4,total=18,updated_at=core.timestamp()))
 arch.commit('research: freeze matched-input multiview controls and 18 development-only jobs')
 print('Prepared 18 multiview-study jobs',flush=True)


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
 collected=[]
 def hook(module,inputs,output):collected.append(dict(module.last_stats))
 if model.is_tiled:
  model.pool.capture=True
  handle=model.pool.register_forward_hook(hook)
 p,y,idx=core.infer(model,core.loader(j['score'],False,seed=j['seed']))
 if model.is_tiled:handle.remove();model.pool.capture=False
 core.save_csv(out/'validation_predictions.csv',core.prediction_rows(p,y,idx))
 diag=dict(note='Off uses the same fitted global encoder at matched preprocessing, not a retrained ablation.')
 for key in ['local_share','local_attention_entropy','descriptor_shift']:
  v=[v for batch in collected for v in batch.get(key,[])]
  if v:diag[key]=dict(mean=float(np.mean(v)),quantiles=np.percentile(v,[0,25,50,75,100]).tolist())
 if model.is_tiled:
  attention=[v for batch in collected for v in batch['local_attention']]
  core.save_csv(out/'view_attention.csv',[dict(index=i,**{f'tile{k}':float(a[k]) for k in range(4)}) for i,a in zip(idx,attention)])
  model.local_enabled=False
  po,yo,idxo=core.infer(model,core.loader(j['score'],False,seed=j['seed']))
  assert idx==idxo and np.array_equal(y,yo)
  diag['off_same_checkpoint_metrics']=core.metrics(po,yo)
  diag['off_prediction_disagreement']=float(np.mean(p.argmax(1)!=po.argmax(1)))
 core.save_json(out/'validation_complete.json',dict(signature=core.signature(j),job_id=job_id,
     recipe=j['recipe'],variant=j['variant'],seed=j['seed'],n=len(y),checkpoint_sha256=completion['checkpoint_sha256'],
     completed_at=core.timestamp(),parameters=sum(v.numel() for v in model.parameters()),
     modifier_parameters=sum(v.numel() for name,v in model.named_parameters() if not name.startswith('backbone.') or '.local_bn.' in name),
     diagnostics=diag,**core.metrics(p,y)))


def check():
 import numpy as np
 import torch
 import torch.nn.functional as F
 from torch.utils.flop_counter import FlopCounterMode
 from models.optimize.cle_multiview import DomainBatchNorm,global_view,local_views
 cfg=verify();core.set_seed(42)
 assert core.digest(NEWCACHE/'resized512_uint8.npy')==cfg['view_cache_sha256']
 # Reassemble all four views to establish exact input coverage/order.
 x=torch.arange(448*448).reshape(1,1,448,448).float()
 tiles=local_views(x)
 rebuilt=torch.cat([torch.cat([tiles[:,0],tiles[:,1]],-1),torch.cat([tiles[:,2],tiles[:,3]],-1)],-2)
 assert torch.equal(x,rebuilt)
 base=arch.ORIGINAL_MODEL(42).eval()
 base_state={k:v.cpu().clone() for k,v in base.state_dict().items()}
 del base
 job=core.read_json(STUDY/'final_jobs.json')[0]
 batch,labels,_,_=next(iter(core.loader(job['train'][:32],False,seed=42)))
 batch=batch.cuda();labels=labels.cuda();budget=[]
 for variant in VARIANTS:
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  model=model_for(42,variant).eval()
  canonical=model.canonical_backbone_state()
  assert canonical.keys()==base_state.keys() and all(torch.equal(v.cpu(),base_state[k]) for k,v in canonical.items())
  if model.is_tiled:
   for m in model.modules():
    if isinstance(m,DomainBatchNorm):assert all(torch.equal(v,m.local_bn.state_dict()[k]) for k,v in m.global_bn.state_dict().items())
   model.local_enabled=False
   with torch.inference_mode():assert torch.equal(model(batch),model.backbone(global_view(batch)))
   model.local_enabled=True
  with torch.inference_mode(),FlopCounterMode(display=False) as counter:model(batch[:1])
  counted_flops=counter.get_total_flops();features=[]
  if model.is_tiled:
   def retain(module,inputs):
    for value in inputs:value.retain_grad();features.append(value)
   handle=model.pool.register_forward_pre_hook(retain)
   model.pool.capture=True
  with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model.train()(batch),labels)
  loss.backward();assert torch.isfinite(loss)
  assert torch.isfinite(model.backbone.conv1.weight.grad).all() and model.backbone.conv1.weight.grad.abs().sum()>0
  if model.is_tiled:
   handle.remove()
   assert all(torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in features)
   a=model.pool.last_attention
   assert torch.allclose(a.sum(1),torch.ones(len(a),device='cuda'),atol=1e-6)
   assert max(model.pool.last_stats['local_share'])<=.5
   bn=next(m for m in model.modules() if isinstance(m,DomainBatchNorm))
   assert not torch.equal(bn.global_bn.running_mean,bn.local_bn.running_mean)
   assert bn.local_bn.weight.grad.abs().sum()>0 and bn.global_bn.weight.grad.abs().sum()>0
   for name,param in model.pool.named_parameters():
    if name.startswith(('query.','key.','score.')):assert param.grad is not None and param.grad.abs().sum()>0
  budget.append(dict(variant=variant,parameters=sum(p.numel() for p in model.parameters()),
                     extra_parameters=sum(p.numel() for name,p in model.named_parameters() if not name.startswith('backbone.') or '.local_bn.' in name),
                     conv_matmul_flops=counted_flops,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))
  del features,canonical
  del model;torch.cuda.empty_cache()
 full=next(r for r in budget if r['variant']=='tile_context');control=next(r for r in budget if r['variant']=='tile_attention')
 assert abs(full['extra_parameters']/control['extra_parameters']-1)<.01
 assert abs(full['conv_matmul_flops']/control['conv_matmul_flops']-1)<.01
 free,_=torch.cuda.mem_get_info()
 assert max(r['peak_allocated_gib'] for r in budget)*1024**3*2<free*.85,'Reduce concurrency after documented resource review, before training'
 core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),budget=budget,
      checks=['complete non-test cache SHA256','four tiles reassemble exact full view',
              'paired canonical backbone and head','local-disabled equals global backbone on same downsampled view',
              'finite real-training-image AMP backward with gradients to global and all local descriptors',
              'independent global/local BN statistics and gradients','query/key/attention gradients and positive normalized weights',
              'full/attention extra parameters and counted FLOPs within 1%','two-job memory headroom'],
      budget_limit='Counted FLOPs omit some pooling, normalization and elementwise costs; no zero-inference-cost claim.'))
 arch.commit('test: verify multiview input coverage, domain normalization, gradients and budgets')
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
 comparisons={v+'_vs_legacy_gap':compare('gap',v) for v in VARIANTS}
 comparisons.update({v+'_vs_matched_global':compare('global224',v) for v in VARIANTS})
 for v in VARIANTS:
  if v!='tile_context':comparisons['cle_vs_'+v]=compare(v,'tile_context')
 summary=[]
 for v in VARIANTS:
  rr=[r for r in records if r['variant']==v]
  summary.append(dict(variant=v,mean_validation_f1=float(np.mean([r['macro_f1'] for r in rr])),
    seed_sd=float(np.std([r['macro_f1'] for r in rr],ddof=1)),mean_malignant_f1=float(np.mean([r['class_f1'][1] for r in rr])),
    modifier_parameters=rr[0]['modifier_parameters'],**comparisons[v+'_vs_matched_global']))
 lines=['# CLE round 4: original-pixel global/local views','',
        'Exploratory development-selected scores; intervals condition on fitted models, without model-selection/multiplicity correction. No test evaluation.','',
        'Primary reference is the newly trained global224 control with matched 512/448 preprocessing. Legacy baseline comparisons are contextual only.','',
        '| Variant | Mean val F1 | Malignant F1 | Delta vs matched global | Patient interval | Extra parameters |',
        '|---|---:|---:|---:|---|---:|']
 for r in summary:lines.append(f'| {r["variant"]} | {r["mean_validation_f1"]:.6f} | {r["mean_malignant_f1"]:.6f} | {r["mean_delta"]:+.6f} | {r["patient_cluster95"]} | {r["modifier_parameters"]} |')
 core.save_json(STUDY/'round4_results.json',records);core.save_json(STUDY/'round4_summary.json',summary)
 core.save_json(STUDY/'round4_comparisons.json',comparisons)
 (STUDY/'ROUND4_REPORT.md').write_text('\n'.join(lines))
 arch.commit('results: summarize matched-input multiview mechanisms on complete development patients')
 core.save_json(STUDY/'status.json',dict(phase='round4_complete_needs_analysis',round=4,total=18,completed=18,updated_at=core.timestamp()))


arch.summarize=summarize
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','check','worker','run']);p.add_argument('--job');a=p.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='check':check()
 elif a.command=='worker':worker(a.job)
 else:arch.run()

