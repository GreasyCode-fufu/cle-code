"""Independent round11 audit: original-patient OOF, inference only."""
from pathlib import Path
import collections
import sys
import numpy as np
import torch
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_dino_oof_study as study
from pretrained_representations import dino
core=study.core;cfg=study.verify();s=study.STUDY
records=core.read_json(s/'manifest.json');jobs=core.read_json(s/'final_jobs.json')
original=core.read_json(study.DATA/'oof_jobs.json');assert len(jobs)==15
assert core.digest(core.CACHE/'resized_uint8.npy')==core.read_json(core.CACHE/'meta.json')['cache_sha256']
base=dino(ROOT);initial={k:v.detach().clone() for k,v in base.state_dict().items()};del base
merged=collections.defaultdict(list);models=[];curves=[];reference_models=[]

def inspect_csv(path,indices):
    rr,metric=study.validate_predictions(path,indices,records)
    y=np.array([int(row['label']) for row in rr]);pred=np.array([int(row['pred']) for row in rr])
    assert abs(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)-metric['macro_f1'])<1e-12
    assert confusion_matrix(y,pred,labels=[0,1,2]).tolist()==metric['confusion_matrix']
    return rr,metric

for job in jobs:
    assert study.done(job)
    old=next(j for j in original if j['id']==job['source_oof_job_id'])
    for name in ['train','select','score','seed','fold','epochs']:assert job[name]==old[name]
    assert not job.get('sample_weights') and job['phase']=='oof'
    assert all(0<=i<2386 for name in ['train','select','score'] for i in job[name])
    core.validate_split(records,job['train'],job['select'],job['score'])
    out=s/'runs'/job['id'];comp=core.read_json(out/'complete.json');outer=core.read_json(out/'outer_complete.json')
    assert comp['signature']==outer['signature']==core.signature(job)
    checkpoint=ROOT/comp['checkpoint'];assert core.digest(checkpoint)==comp['checkpoint_sha256']==outer['checkpoint_sha256']
    log=study.arch.read_csv(out/'train_log.csv')
    assert [int(row['epoch']) for row in log]==list(range(60)) and comp['epochs_completed']==60
    assert all(np.isfinite(float(v)) for row in log for v in row.values())
    best=max(log,key=lambda row:float(row['val_macro_f1']))
    assert int(best['epoch'])==comp['best_epoch'] and abs(float(best['val_macro_f1'])-comp['best_select_f1'])<1e-12
    inner,mi=inspect_csv(out/'inner_predictions.csv',job['select'])
    rr,mo=inspect_csv(out/'oof_predictions.csv',job['score'])
    assert abs(mi['macro_f1']-comp['best_select_f1'])<1e-12
    assert abs(mo['macro_f1']-outer['macro_f1'])<1e-12 and mo['confusion_matrix']==outer['confusion_matrix']
    model=study.transfer.model_for(job['seed'],'full_finetune')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);assert all(torch.isfinite(v).all() for v in state.values())
    model.load_state_dict(state,strict=True)
    assert sum(p.numel() for p in model.parameters())==21631107==comp['parameters']==comp['trainable_parameters']
    source=model.source_state();squared=count=0
    for name,value in source.items():
        squared+=float((value.detach().cpu().double()-initial[name].double()).square().sum());count+=value.numel()
    rms=(squared/count)**.5;assert rms>0
    differences={}
    for name,ids,expected in [('inner',job['select'],inner),('outer',job['score'],rr)]:
        p,y,idx=core.infer(model,core.loader(ids,False,seed=job['seed']))
        order=np.argsort(idx);p=p[order];y=y[order];idx=np.asarray(idx)[order].tolist()
        saved=np.array([[float(row[f'p{k}']) for k in range(3)] for row in expected])
        diff=float(np.abs(p-saved).max());assert diff<2e-6
        assert idx==[int(row['index']) for row in expected] and np.array_equal(p.argmax(1),saved.argmax(1))
        assert np.array_equal(y,[int(row['label']) for row in expected]);differences[name]=diff
    assert checkpoint.owner()=='login' and all(f.owner()=='login' for f in out.rglob('*') if f.is_file())
    merged['dino_full',job['seed']]+=rr
    models.append(dict(job=job['id'],checkpoint_sha256=comp['checkpoint_sha256'],inner_n=len(inner),outer_n=len(rr),
                       best_inner_epoch=comp['best_epoch'],source_rms_drift=rms,prediction_max_diffs=differences))
    curves.append(dict(job=job['id'],seed=job['seed'],fold=job['fold'],best_epoch=comp['best_epoch'],best_inner_f1=mi['macro_f1'],
                       last_inner_f1=float(log[-1]['val_macro_f1']),last_train_accuracy=float(log[-1]['train_acc']),outer_f1=mo['macro_f1']))
    del model,state,source;torch.cuda.empty_cache();print('audited',job['id'],flush=True)

for entry in core.read_json(s/'reference_index.json'):
    old=entry['source_job'];out=s/'references'/entry['arm']/f'oof_s{old["seed"]}_f{old["fold"]}'
    source=ROOT/entry['source_dir'];comp=core.read_json(out/'complete.json')
    assert comp['signature']==entry['source_signature']==study.source_signature(old,source)
    checkpoint=ROOT/comp['checkpoint'];assert core.digest(checkpoint)==comp['checkpoint_sha256']==entry['checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);assert all(torch.isfinite(v).all() for v in state.values());del state
    log=study.arch.read_csv(out/'train_log.csv');assert [int(row['epoch']) for row in log]==list(range(60))
    assert comp['best_epoch']==int(max(log,key=lambda row:float(row['val_macro_f1']))['epoch'])
    rr,metric=inspect_csv(out/'oof_predictions.csv',old['score'])
    assert abs(metric['macro_f1']-comp['outer_metrics']['macro_f1'])<1e-12
    core.validate_split(records,old['train'],old['select'],old['score'])
    merged[entry['arm'],old['seed']]+=rr
    reference_models.append(dict(arm=entry['arm'],seed=old['seed'],fold=old['fold'],checkpoint_sha256=comp['checkpoint_sha256']))
    assert checkpoint.owner()=='login'
for seed in cfg['seeds']:
    path=s/'references'/f'dino_frozen_ridge_s{seed}.csv'
    assert core.digest(path)==core.digest(study.REP/'oof'/f'dino_concat_s{seed}.csv')
    merged['dino_frozen_ridge',seed]=inspect_csv(path,range(2386))[0]

groups=sorted({record['group_id'] for record in records if record['split']=='train'});assert len(groups)==185
weights=np.random.default_rng(20261024).multinomial(185,np.ones(185)/185,size=20000)
matrices={};results=core.read_json(s/'round11_results.json')
for (arm,seed),rr in merged.items():
    rr=sorted(rr,key=lambda row:int(row['index']));assert [int(row['index']) for row in rr]==list(range(2386))
    assert rr==study.arch.read_csv(s/'oof'/f'{arm}_s{seed}.csv')
    y=np.array([int(row['label']) for row in rr]);pred=np.array([int(row['pred']) for row in rr])
    r=next(v for v in results if v['scheme']==arm and v['seed']==seed)
    cm=confusion_matrix(y,pred,labels=[0,1,2]);score=f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)
    assert abs(score-r['macro_f1'])<1e-12 and cm.tolist()==r['confusion_matrix']
    matrix=np.zeros((185,3,3));np.add.at(matrix,([groups.index(row['group_id']) for row in rr],y,pred),1)
    assert abs(np.mean(np.trace(matrix,axis1=-2,axis2=-1)/matrix.sum((1,2)))-r['patient_mean_accuracy'])<1e-12
    assert abs(cm[1,1]/cm[1].sum()-r['malignant_recall'])<1e-12
    matrices[arm,seed]=matrix
assert len(matrices)==12
def f1(m):
    den=m.sum(-1)+m.sum(-2)
    return np.mean(np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),den,out=np.zeros_like(den),where=den!=0),axis=-1)
comp=core.read_json(s/'paired_comparisons.json');gates=[]
for arm in study.REFS:
    ds=[];boot=[]
    for seed in cfg['seeds']:
        a,b=matrices[arm,seed],matrices['dino_full',seed]
        ds.append(float(f1(b.sum(0))-f1(a.sum(0))))
        boot.append(f1(np.einsum('bg,gij->bij',weights,b))-f1(np.einsum('bg,gij->bij',weights,a)))
    distribution=np.mean(boot,axis=0);family=np.percentile(distribution,[2.5/3,100-2.5/3])
    r=comp['comparisons']['dino_full_vs_'+arm]
    assert np.allclose(ds,r['seed_deltas'],atol=1e-12)
    assert np.allclose(np.percentile(distribution,[2.5,97.5]),r['patient95'],atol=1e-12)
    assert np.allclose(family,r['family_interval'],atol=1e-12)
    gate=min(ds)>0 and np.mean(ds)>=.01 and family[0]>0
    assert bool(gate)==r['strong_pair'];gates.append(gate)
assert bool(all(gates))==comp['strong_primary_followup']
assert all(f.owner()=='login' for f in s.rglob('*') if f.is_file())
core.save_json(s/'training_curve_audit.json',curves)
core.save_json(s/'completion_audit.json',dict(passed=True,timestamp=core.timestamp(),new_models=15,reused_cnn_models=30,
    frozen_dino_reference_files=3,merged_oof_files=12,
    checks=['immutable source/config/cache hashes and exact original15 patient splits','only original train, no grades/weights, three-way patient isolation',
            '15 new signed checkpoints,60ep,inner-best selection,finite strict21631107-parameter models and source drift',
            'all new inner and outer probabilities independently reproduced','30 reused CNN signatures/SHA/60ep/outer metrics',
            'three frozen-DINO references identical to audited R9','12 full2386 OOF sets, each image three predictions per scheme',
            'independent pooled F1/confusion/malignant recall/patient-equal accuracy','185-patient paired bootstrap,three family intervals and full gate','login ownership'],
    models=models,reference_models=reference_models,comparisons=comp))
print('PASS independent round11 audit',flush=True)
