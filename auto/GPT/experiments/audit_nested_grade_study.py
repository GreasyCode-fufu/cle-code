"""Independent R14 audit: saved linear heads, patient exclusions and all CNN predictions."""
from pathlib import Path
import collections
import json
import sys
import numpy as np
import torch
from scipy.special import softmax, logsumexp
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_nested_grade_study as study
from pretrained_representations import state_sha
core=study.core;arch=study.arch;s=study.STUDY;cfg=study.verify();grading=study.verify_grading()
records=core.read_json(s/'manifest.json');jobs=core.read_json(s/'final_jobs.json');units=core.read_json(s/'grading_jobs.json')
original=core.read_json(study.DATA/'oof_jobs.json');index=core.read_json(s/'baseline_index.json')
assert len(jobs)==60 and len(units)==15 and cfg['new_neural_trainings']==60 and cfg['aliases']==[]
assert core.digest(core.CACHE/'resized_uint8.npy')==core.read_json(core.CACHE/'meta.json')['cache_sha256']
assert core.digest(study.FEATURES)==grading['feature_sha256']
with np.load(study.FEATURES,allow_pickle=False) as data:
    assert data['indices'].tolist()==list(range(2386)) and data['features'].shape==(2386,1280)
    features=data['features'][:,512:1280].astype(np.float64)
assert np.isfinite(features).all()
labels=np.array([r['label'] for r in records]);groups=np.array([r['group_id'] for r in records])
grade_audit=[];per_class_aggregate=collections.defaultdict(collections.Counter)
head_count=0
for unit in units:
    assert study.grade_done(unit)
    old=next(o for o in original if o['id']==unit['source_job_id'])
    for key in ['seed','fold','train','select','score','epochs']:assert unit[key]==old[key]
    core.validate_split(records,old['train'],old['select'],old['score'])
    assert all(records[i]['split']=='train' and 0<=i<2386 for key in ['train','select','score'] for i in old[key])
    out=s/'grades'/unit['id'];manifest=core.read_json(out/'head_manifest.json');assert len(manifest['heads'])==15
    saved_cross={(int(r['replicate']),int(r['index'])):r for r in arch.read_csv(out/'crossfit_predictions.csv')}
    assert len(saved_cross)==3*len(old['train'])
    predictions=collections.defaultdict(list);max_residual=0.;objectives=[];iterations=[]
    active=np.array(old['train'])
    for replicate in [42,43,44]:
        splitter=StratifiedGroupKFold(5,shuffle=True,random_state=old['seed']*100000+old['fold']*100+replicate)
        visited=[]
        for fold,(a,b) in enumerate(splitter.split(active,labels[active],groups[active])):
            fit,held=active[a],active[b];h=manifest['heads'][f'r{replicate}_f{fold}'];path=ROOT/h['path']
            assert core.digest(path)==h['sha256'] and h['fit_indices']==fit.tolist() and h['held_indices']==held.tolist()
            assert set(groups[fit]).isdisjoint(groups[held]) and set(fit)|set(held)==set(old['train'])
            assert (set(groups[fit])|set(groups[held])).isdisjoint(set(groups[old['select']])|set(groups[old['score']]))
            with np.load(path,allow_pickle=False) as head:
                assert head['classes'].tolist()==[0,1,2] and head['fit_indices'].tolist()==fit.tolist() and head['held_indices'].tolist()==held.tolist()
                assert all(np.isfinite(head[k]).all() for k in head.files)
                mean=features[fit].mean(0);scale=np.maximum(features[fit].std(0),1e-6)
                assert np.array_equal(head['mean'],mean) and np.array_equal(head['scale'],scale)
                x=(features[fit]-mean)/scale;coef=head['coef'];intercept=head['intercept']
                z=x@coef.T+intercept;p=softmax(z,axis=1);residual=p-np.eye(3)[labels[fit]]
                maximum=max(float(np.abs(x.T@residual/len(fit)+coef.T/len(fit)).max()),float(np.abs(residual.mean(0)).max()))
                assert maximum<5e-5 and abs(maximum-float(head['gradient_max']))<1e-10
                assert int(head['n_iter'].max())<2000
                objective=float((logsumexp(z,axis=1)-z[np.arange(len(fit)),labels[fit]]).mean()+np.square(coef).sum()/(2*len(fit)))
                hp=softmax(((features[held]-mean)/scale)@coef.T+intercept,axis=1)
                for i,prob in zip(held,hp):
                    row=saved_cross[replicate,int(i)]
                    assert int(row['nested_fold'])==fold and int(row['label'])==labels[i]
                    assert np.max(np.abs(prob-np.array([float(row[f'p{k}']) for k in range(3)])))<1e-10
                    assert int(row['pred'])==prob.argmax()
                    predictions[int(i)].append((replicate,prob))
                max_residual=max(max_residual,maximum);objectives.append(objective);iterations.append(int(head['n_iter'].max()))
            assert path.owner()=='login';visited+=held.tolist();head_count+=1
        assert sorted(visited)==sorted(old['train'])
    grade_rows=arch.read_csv(out/'grades.csv');assert [int(r['index']) for r in grade_rows]==old['train']
    grades={}
    for row in grade_rows:
        i=int(row['index']);parts=sorted(predictions[i],key=lambda x:x[0]);assert [p[0] for p in parts]==[42,43,44]
        prob=np.array([p[1] for p in parts]);correct=float((prob.argmax(1)==labels[i]).mean());confidence=float(prob[:,labels[i]].mean())
        if correct>=.95 and confidence>=.95:g='A'
        elif correct>=.85 and confidence>=.85:g='B'
        elif correct<.30 or confidence<.30:g='E'
        elif correct<.60 or confidence<.60:g='D'
        else:g='C'
        assert row['grade']==g and abs(float(row['confidence'])-confidence)<1e-10 and abs(float(row['correct_rate'])-correct)<1e-12
        assert row['group_id']==groups[i] and row['sample_id']==records[i]['sample_id'] and int(row['label'])==labels[i]
        grades[i]=g;per_class_aggregate[int(labels[i])][g]+=1
    base_weights={'A':1.,'B':1.,'C':.75,'D':.25,'E':.1}
    w=np.array([base_weights[grades[int(i)]] for i in active]);shuffle=w.copy();rng=np.random.default_rng(20261024+old['seed']*100+old['fold'])
    ac=sorted(int(i) for i in active if grades[int(i)] in ['A','B','C']);random_ac=[]
    for label in [0,1,2]:
        positions=np.flatnonzero(labels[active]==label);w[positions]/=w[positions].mean();shuffle[positions]=rng.permutation(w[positions])
        n=sum(labels[i]==label for i in ac);random_ac+=rng.choice(active[positions],size=n,replace=False).tolist()
        assert np.array_equal(np.sort(w[positions]),np.sort(shuffle[positions])) and abs(w[positions].mean()-1)<1e-12
    expected={'grade_weighted':(active.tolist(),dict(zip(map(str,active),map(float,w)))),
              'shuffled_weighted':(active.tolist(),dict(zip(map(str,active),map(float,shuffle)))),
              'ac_only':(ac,{}),'random_ac':(sorted(random_ac),{})}
    plans=core.read_json(out/'student_plans.json');effective=[]
    for arm,(ids,weights) in expected.items():
        j=next(j for j in jobs if j['arm']==arm and j['seed']==old['seed'] and j['fold']==old['fold'])
        assert j['train']==ids and j['sample_weights']==weights and plans[arm]==dict(train=ids,sample_weights=weights)
        for key in ['select','score']:assert j[key]==old[key]
        assert j['grade_complete_sha256']==core.digest(out/'complete.json')
        effective.append(json.dumps([ids,[weights.get(str(i),1.) for i in ids]]))
    assert len(set(effective))==4
    grade_audit.append(dict(unit=unit['id'],fit_n=len(active),nested_heads=15,maximum_gradient_residual=max_residual,
                     max_iterations=max(iterations),head_objective_range=[min(objectives),max(objectives)],
                     original_per_class=[int((labels[active]==c).sum()) for c in range(3)],
                     ac_per_class=[int((labels[ac]==c).sum()) for c in range(3)]))
    print('audited grading',unit['id'],flush=True)
assert head_count==225

initial_model=arch.ORIGINAL_MODEL(42).cpu();initial={k:v.detach().clone() for k,v in initial_model.state_dict().items()};del initial_model
units_audit=[];curves=[];merged=collections.defaultdict(list)
model_items=[('full_ce',e['source_job'],s/'baselines'/e['source_job']['id']) for e in index]
model_items += [(j['arm'],j,s/'runs'/j['id']) for j in jobs]
for arm,job,out in model_items:
    if arm=='full_ce':
        old=next(o for o in original if o['id']==job['id']);assert job==old
        expected_signature=study.original_signature(job)
    else:
        assert study.student_done(job) and 'alias' not in job
        expected_signature=core.signature(job)
    core.validate_split(records,job['train'],job['select'],job['score'])
    assert all(records[i]['split']=='train' and 0<=i<2386 for key in ['train','select','score'] for i in job[key])
    comp=core.read_json(out/'complete.json');meta=core.read_json(out/'run_meta.json');checkpoint=ROOT/comp['checkpoint']
    assert comp['signature']==meta['signature']==expected_signature and core.digest(checkpoint)==comp['checkpoint_sha256']
    assert comp['pretrained_sha256']==grading['pretrained_sha256'] and meta['train_count']==len(job['train'])
    log=arch.read_csv(out/'train_log.csv');assert [int(r['epoch']) for r in log]==list(range(60)) and comp['epochs_completed']==60
    assert all(np.isfinite(float(v)) for row in log for v in row.values())
    best=max(log,key=lambda row:float(row['val_macro_f1']))
    assert comp['best_epoch']==int(best['epoch']) and abs(comp['best_select_f1']-float(best['val_macro_f1']))<1e-12
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert set(state)==set(initial) and all(torch.isfinite(value).all() for value in state.values())
    model=arch.ORIGINAL_MODEL(job['seed']);model.load_state_dict(state,strict=True)
    assert sum(p.numel() for p in model.parameters())==21286211
    bn_names=[name for name,m in model.named_modules() if isinstance(m,torch.nn.BatchNorm2d)];assert len(bn_names)==36
    assert all(state[name+'.num_batches_tracked']>initial[name+'.num_batches_tracked'] for name in bn_names)
    nonfc=[key for key in state if not key.startswith('fc.')];squares=sum(float((state[k].double()-initial[k].double()).square().sum()) for k in nonfc)
    rms=(squares/sum(state[k].numel() for k in nonfc))**.5;assert rms>0
    model.eval().requires_grad_(False);before=state_sha(model)
    inner,y,ids=core.infer(model,core.loader(job['select'],False,seed=job['seed']))
    assert ids==job['select'] and abs(f1_score(y,inner.argmax(1),labels=[0,1,2],average='macro',zero_division=0)-comp['best_select_f1'])<1e-12
    prob,y,ids=core.infer(model,core.loader(job['score'],False,seed=job['seed']))
    assert ids==job['score'] and state_sha(model)==before
    rows=arch.read_csv(out/'oof_predictions.csv');assert [int(r['index']) for r in rows]==job['score']
    saved=np.array([[float(row[f'p{k}']) for k in range(3)] for row in rows]);difference=float(np.abs(saved-prob).max())
    assert difference<2e-6 and np.array_equal(prob.argmax(1),[int(r['pred']) for r in rows])
    score=f1_score(y,prob.argmax(1),labels=[0,1,2],average='macro',zero_division=0)
    assert abs(score-comp['outer_metrics']['macro_f1'])<1e-12
    assert confusion_matrix(y,prob.argmax(1),labels=[0,1,2]).tolist()==comp['outer_metrics']['confusion_matrix']
    for row in rows:
        m=records[int(row['index'])];assert row['group_id']==m['group_id'] and int(row['label'])==m['label'] and row['sample_id']==m['sample_id']
    assert checkpoint.owner()=='login' and all(p.owner()=='login' for p in out.rglob('*') if p.is_file())
    merged[arm,job['seed']]+=rows
    units_audit.append(dict(arm=arm,job=job['id'],checkpoint_sha256=comp['checkpoint_sha256'],source_rms_drift=rms,prediction_max_difference=difference,outer_f1=score))
    curves.append(dict(arm=arm,job=job['id'],seed=job['seed'],fold=job['fold'],train_n=len(job['train']),best_epoch=comp['best_epoch'],
                       best_inner_f1=comp['best_select_f1'],last_inner_f1=float(log[-1]['val_macro_f1']),last_train_accuracy=float(log[-1]['train_acc']),outer_f1=score))
    del model,state;torch.cuda.empty_cache();print('audited neural',arm,job['id'],flush=True)

patient_groups=sorted(set(groups[:2386]));assert len(patient_groups)==185
results=core.read_json(s/'round14_results.json');matrices={};predictions={}
for (arm,seed),rows in merged.items():
    rows.sort(key=lambda r:int(r['index']));assert [int(r['index']) for r in rows]==list(range(2386))
    assert arch.read_csv(s/'oof'/f'{arm}_s{seed}.csv')==rows
    p=np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows]);pred=p.argmax(1);y=labels[:2386]
    r=next(r for r in results if r['variant']==arm and r['seed']==seed)
    cm=confusion_matrix(y,pred,labels=[0,1,2]);score=f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)
    assert abs(score-r['macro_f1'])<1e-12 and cm.tolist()==r['confusion_matrix']
    m=np.zeros((185,3,3));np.add.at(m,([patient_groups.index(g) for g in groups[:2386]],y,pred),1)
    assert abs(np.mean(np.trace(m,axis1=-2,axis2=-1)/m.sum((1,2)))-r['patient_mean_accuracy'])<1e-12
    assert abs(cm[1,1]/cm[1].sum()-r['malignant_recall'])<1e-12
    matrices[arm,seed]=m;predictions[arm,seed]=pred
draws=np.random.default_rng(20261024).multinomial(185,np.ones(185)/185,size=20000)
def f1(m):
    d=m.sum(-1)+m.sum(-2)
    return np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),d,out=np.zeros_like(d),where=d!=0).mean(-1)
stored=core.read_json(s/'paired_comparisons.json');gates=[]
for a,b in [('grade_weighted','full_ce'),('grade_weighted','shuffled_weighted'),('ac_only','full_ce'),('ac_only','random_ac')]:
    ds=[];boot=[]
    for seed in [42,43,44]:
        ma,mb=matrices[a,seed],matrices[b,seed];ds.append(float(f1(ma.sum(0))-f1(mb.sum(0))))
        boot.append(f1(np.einsum('bg,gij->bij',draws,ma))-f1(np.einsum('bg,gij->bij',draws,mb)))
    distribution=np.mean(boot,axis=0);family=np.percentile(distribution,[.625,99.375]);r=next(r for r in stored['comparisons'] if r['candidate']==a and r['reference']==b)
    assert np.allclose(ds,r['seed_deltas'],atol=1e-12) and np.allclose(np.percentile(distribution,[2.5,97.5]),r['patient_cluster95'],atol=1e-12)
    assert np.allclose(family,r['family_interval'],atol=1e-12)
    gate=bool(min(ds)>0 and np.mean(ds)>=.01 and family[0]>0);assert gate==r['strong'];gates.append(gate)
assert all(gates[:2])==stored['strong_primary_followup'] and all(gates[2:])==stored['strong_secondary_followup']
assert len(matrices)==15 and all(p.owner()=='login' for p in s.rglob('*') if p.is_file())
core.save_json(s/'training_curve_audit.json',curves)
core.save_json(s/'grade_audit.json',dict(units=grade_audit,head_count=head_count,per_class_grade_assignments={str(k):dict(v) for k,v in per_class_aggregate.items()},
    note='Counts aggregate repeated fit appearances across15 outer models, not independent images.'))
core.save_json(s/'completion_audit.json',dict(passed=True,timestamp=core.timestamp(),new_neural_trainings=60,reused_neural_models=15,convex_grader_fits=225,merged_oof_files=15,
    checks=['dual frozen configs, original patient folds and original-train-only access', '225 saved heads fit-only transforms, patient exclusion and convex gradient residuals',
            'three held-patient predictions per fit image and independent legacy A-E rules', 'exact normalized/shuffled class weight multisets and AC/random class counts',
            '75 signed CNN checkpoints,60ep,inner-only best,finite parameters,ordinary BN and source drift',
            'all75 inner and outer inference reproduced with unchanged eval state', '15 full2386 OOF and independent metrics/four family intervals/complete gates', 'login ownership'],
    neural_units=units_audit,grading_units=grade_audit,comparisons=stored))
print('PASS independent round14 audit',flush=True)
