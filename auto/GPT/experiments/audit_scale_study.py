"""Audit frozen round 3 without retraining or new model inference."""
import collections
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_scale_study as study
core=study.core
cfg=study.verify()
records=core.read_json(study.STUDY/'manifest.json')
jobs=core.read_json(study.STUDY/'final_jobs.json')
assert len(jobs)==18
def rows(path):
 with open(path) as f:return list(csv.DictReader(f))

audit=[]
for job in jobs:
 assert study.arch.done(job)
 out=study.STUDY/'runs'/job['id']
 result=core.read_json(out/'validation_complete.json')
 comp=core.read_json(out/'complete.json')
 assert comp['checkpoint_sha256']==result['checkpoint_sha256']
 pred=rows(out/'validation_predictions.csv')
 assert len(pred)==538 and len({r['index'] for r in pred})==538
 assert {int(r['index']) for r in pred}==set(cfg['val_indices'])
 p=np.array([[float(r['p'+str(k)]) for k in range(3)] for r in pred])
 y=np.array([int(r['label']) for r in pred])
 assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=2e-6)
 assert np.array_equal(p.argmax(1),[int(r['pred']) for r in pred])
 for r in pred:
  m=records[int(r['index'])]
  assert r['sample_id']==m['sample_id'] and r['group_id']==m['group_id'] and int(r['label'])==m['label']
 metrics=core.metrics(p,y)
 assert abs(metrics['macro_f1']-result['macro_f1'])<1e-12
 assert metrics['confusion_matrix']==result['confusion_matrix']
 log=rows(out/'train_log.csv')
 assert [int(r['epoch']) for r in log]==list(range(60))
 assert set(job['score'])==set(cfg['val_indices']) and set(job['select'])==set(cfg['val_indices'])
 assert {records[i]['split'] for i in job['train']}=={'train'}
 assert not set(job['train']) & set(cfg['test_indices'])
 assert not {records[i]['group_id'] for i in job['train']} & {records[i]['group_id'] for i in job['score']}
 assert all(p.owner()=='login' for p in out.rglob('*') if p.is_file())
 assert (ROOT/comp['checkpoint']).owner()=='login'
 audit.append(dict(job=job['id'],epochs=60,validation_images=538,checkpoint_sha256=comp['checkpoint_sha256'],macro_f1=result['macro_f1']))

baseline=[]
for seed in [42,43,44]:
 baseline.append(rows(study.STUDY/'baselines'/f'grade_weighted_s{seed}'/'validation_predictions.csv'))
by_index={int(r['index']):dict(r,wrong_votes=0) for r in baseline[0]}
for rr in baseline:
 for r in rr:by_index[int(r['index'])]['wrong_votes']+=int(r['pred']!=r['label'])
by_subtype=collections.defaultdict(list);by_group=collections.defaultdict(list)
for idx,r in by_index.items():
 m=records[idx];name=Path(m['path']).name
 subtype=name.split('_S_')[0].replace('val_','')
 by_subtype[subtype].append(r);by_group[m['group_id']].append(r)
def summarize(v):
 return dict(n=len(v),mean_error_rate=sum(r['wrong_votes'] for r in v)/(3*len(v)),
             wrong_all_seeds=sum(r['wrong_votes']==3 for r in v),any_seed_wrong=sum(r['wrong_votes']>0 for r in v))
groups=sorted([dict(group_id=k,**summarize(v)) for k,v in by_group.items()],key=lambda r:r['wrong_all_seeds'],reverse=True)
diag=dict(baseline_only=True,validation_groups=len(groups),all_images=summarize(list(by_index.values())),
          subtypes={k:summarize(v) for k,v in by_subtype.items()},groups=groups,
          limitations=['Subtype strings are filename-derived and 12 validation filenames conflict with current directories; not new clinical truth.',
                       'Concentrated errors do not establish incorrect labels; patient groups have unequal image counts.'])
core.save_json(study.STUDY/'completion_audit.json',dict(passed=True,timestamp=core.timestamp(),
 checks=['source and data signatures','18 checkpoint SHA256','18 x 60 epoch logs','18 x full validation 538 predictions',
         'finite normalized probabilities and recomputed metrics','patient disjointness','no test score indices','login ownership'],jobs=audit))
core.save_json(study.STUDY/'error_distribution.json',diag)
print(json.dumps(dict(passed=True,jobs=len(audit),error_summary=diag['all_images'],subtypes=diag['subtypes'],largest_error_groups=groups[:5]),indent=2))

