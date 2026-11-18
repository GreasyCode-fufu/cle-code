"""Independent split-local OOF grading and frozen AC cohort audit."""
from pathlib import Path
import collections
import json
import sys
import numpy as np
import torch
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_ac_dataset as study
from pretrained_representations import state_sha
core=study.core;arch=study.arch;s=study.STUDY;cfg=study.verify()
records=core.read_json(s/'manifest.json');dataset=core.read_json(s/'dataset_config.json');retained=core.read_json(s/'dataset_manifest.json')
assert core.digest(s/'dataset_manifest.json')==dataset['manifest_sha256']
assert core.digest(core.CACHE/'resized_uint8.npy')==core.read_json(core.CACHE/'meta.json')['cache_sha256']
new=core.read_json(s/'final_jobs.json');reused=core.read_json(s/'reused_index.json');assert len(new)==30 and len(reused)==15
items=[(e['source_job'],s/'reused_train'/e['source_job']['id'],True) for e in reused]+[(j,s/'runs'/j['id'],False) for j in new]
predictions=collections.defaultdict(list);units=[]
for job,out,is_old in items:
    expected=study.old_signature(job) if is_old else core.signature(job)
    if not is_old:assert study.grade_done(job)
    core.validate_split(records,job['train'],job['select'],job['score'])
    split=records[job['score'][0]]['split']
    assert {records[i]['split'] for key in ['train','select','score'] for i in job[key]}=={split}
    assert split==('train' if is_old else job['grade_split'])
    comp=core.read_json(out/'complete.json');assert comp['signature']==expected and comp['epochs_completed']==60
    ck=ROOT/comp['checkpoint'];assert core.digest(ck)==comp['checkpoint_sha256']
    log=arch.read_csv(out/'train_log.csv');assert [int(r['epoch']) for r in log]==list(range(60))
    best=max(log,key=lambda r:float(r['val_macro_f1']))
    assert int(best['epoch'])==comp['best_epoch'] and abs(float(best['val_macro_f1'])-comp['best_select_f1'])<1e-12
    state=torch.load(ck,map_location='cpu',weights_only=True);assert all(torch.isfinite(v).all() for v in state.values())
    m=study.new_model(job['seed']);m.load_state_dict(state,strict=True);m.eval().requires_grad_(False)
    assert sum(p.numel() for p in m.parameters())==21286211
    before=state_sha(m);p,y,idx=core.infer(m,core.loader(job['score'],False,seed=job['seed']))
    assert idx==job['score'] and state_sha(m)==before
    rows=arch.read_csv(out/'oof_predictions.csv');assert [int(r['index']) for r in rows]==idx
    saved=np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows]);difference=float(np.abs(saved-p).max())
    assert difference<2e-6 and np.array_equal(p.argmax(1),[int(r['pred']) for r in rows])
    f1=f1_score(y,p.argmax(1),labels=[0,1,2],average='macro',zero_division=0)
    assert abs(f1-comp['outer_metrics']['macro_f1'])<1e-12
    for row in rows:
        i=int(row['index']);assert int(row['label'])==records[i]['label'] and row['group_id']==records[i]['group_id']
        predictions[i].append((job['seed'],np.array([float(row[f'p{k}']) for k in range(3)])))
    assert ck.owner()=='login'
    units.append(dict(job=job['id'],split=split,reused=is_old,checkpoint_sha256=comp['checkpoint_sha256'],max_probability_difference=difference))
    del state,m;torch.cuda.empty_cache();print('audited',job['id'],flush=True)
assert set(predictions)==set(range(3595))
grade_rows={}
for split in ['train','val','test']:
    assert core.digest(s/f'{split}_grades.csv')==dataset['grade_csv_sha256'][split]
    for row in arch.read_csv(s/f'{split}_grades.csv'):grade_rows[int(row['index'])]=row
assert set(grade_rows)==set(range(3595));kept=[];excluded=[]
for record in records:
    i=record['index'];parts=sorted(predictions[i],key=lambda v:v[0]);assert [v[0] for v in parts]==[42,43,44]
    p=np.array([v[1] for v in parts]);c=float((p.argmax(1)==record['label']).mean());q=float(p[:,record['label']].mean())
    if c>=.95 and q>=.95:g='A'
    elif c>=.85 and q>=.85:g='B'
    elif c<.30 or q<.30:g='E'
    elif c<.60 or q<.60:g='D'
    else:g='C'
    row=grade_rows[i];assert row['grade']==g and abs(float(row['confidence'])-q)<1e-10 and abs(float(row['correct_rate'])-c)<1e-12
    assert row['split']==record['split'] and int(row['label'])==record['label'] and row['sha256']==record['sha256']
    assert core.digest(ROOT/record['path'])==record['sha256']
    (kept if g in ['A','B','C'] else excluded).append(i)
assert [r['index'] for r in retained]==kept
assert [int(r['index']) for r in arch.read_csv(s/'excluded_DE.csv')]==excluded
patient_sets=[]
for split in ['train','val','test']:
    rows=[r for r in retained if r['split']==split];ids=[r['index'] for r in rows]
    assert ids==dataset['indices'][split] and len(rows)==dataset['counts'][split]['retained']
    assert {r['label'] for r in rows}=={0,1,2};patient_sets.append({r['group_id'] for r in rows})
    for row in rows:
        original=records[row['index']];link=ROOT/row['ac_path']
        assert row['label']==original['label'] and row['group_id']==original['group_id']
        assert link.is_symlink() and link.resolve()==(ROOT/original['path']).resolve()
    for c in range(3):assert sum(r['label']==c for r in rows)==dataset['counts'][split]['class_retained'][str(c)]
assert all(not patient_sets[a]&patient_sets[b] for a,b in [(0,1),(0,2),(1,2)])
assert all(p.owner()=='login' for p in s.rglob('*') if p.is_file())
core.save_json(s/'independent_dataset_audit.json',dict(passed=True,timestamp=core.timestamp(),new_models=30,reused_models=15,original_images=3595,
    retained_images=len(kept),excluded_images=len(excluded),counts=dataset['counts'],units=units,
    checks=['45 source signatures/checkpoint hashes,60ep and patient-isolated within-split selection',
            'all45 held-patient inference reproduced independently and eval states unchanged','three predictions per original image and unchanged A-E rules',
            'all3595 original image hashes/labels,exact retained and excluded indices','symlink targets,class counts and original train/val/test patient isolation','login ownership']))
print('PASS independent AC dataset audit',flush=True)
