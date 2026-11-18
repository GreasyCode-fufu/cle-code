"""R13 independent audit: new training, patient splits, calibration and mixtures."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.optimize import brentq
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_fusion_confirmation as study
from pretrained_representations import state_sha,dino
core=study.core;cfg=study.verify();s=study.STUDY
records=core.read_json(s/'manifest.json');jobs=core.read_json(s/'final_jobs.json');folds=core.read_json(s/'folds.json')
assert len(jobs)==45 and cfg['seeds']==[45,46,47]
assert core.digest(core.CACHE/'resized_uint8.npy')==core.read_json(core.CACHE/'meta.json')['cache_sha256']
cnn=study.arch.ORIGINAL_MODEL(45).cpu();cnn_initial={k:v.detach().clone() for k,v in cnn.state_dict().items()};del cnn
vit=dino(ROOT);vit_initial={k:v.detach().clone() for k,v in vit.state_dict().items()};del vit

def softmax(z):
    z=np.asarray(z,dtype=np.float64);shift=z-z.max(1,keepdims=True);value=np.exp(shift)
    return value/value.sum(1,keepdims=True)

unit_predictions={};audit=[];curves=[]
for job in jobs:
    assert study.done(job)
    fold=next(f for f in folds if f['seed']==job['seed'] and f['fold']==job['fold'])
    for name in ['train','select','score']:assert job[name]==fold[name]
    assert not job.get('sample_weights') and job['phase']=='oof'
    assert all(0<=i<2386 for key in ['train','select','score'] for i in job[key])
    core.validate_split(records,job['train'],job['select'],job['score'])
    out=s/'runs'/job['id'];comp=core.read_json(out/'complete.json');outer_comp=core.read_json(out/'outer_complete.json')
    assert comp['signature']==outer_comp['signature']==core.signature(job)
    assert core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']==outer_comp['checkpoint_sha256']
    expected_pretrained=cfg['dino_pretrained_sha256'] if job['arm']=='dino_full' else cfg['pretrained_sha256']
    assert comp['pretrained_sha256']==expected_pretrained
    log=study.arch.read_csv(out/'train_log.csv')
    assert [int(row['epoch']) for row in log]==list(range(60)) and comp['epochs_completed']==60
    assert all(np.isfinite(float(v)) for row in log for v in row.values())
    best=max(log,key=lambda row:float(row['val_macro_f1']))
    assert int(best['epoch'])==comp['best_epoch'] and abs(float(best['val_macro_f1'])-comp['best_select_f1'])<1e-12
    meta=core.read_json(study.CACHE/(job['id']+'.json'));path=study.CACHE/(job['id']+'.npz')
    assert meta['signature']==core.signature(job) and core.digest(path)==meta['sha256']==outer_comp['cache_sha256']
    assert meta['effective_indices']==job['select']+job['score'] and meta['state_unchanged']
    with np.load(path,allow_pickle=False) as data:arrays={key:data[key] for key in data.files}
    assert arrays['inner_indices'].tolist()==job['select'] and arrays['score_indices'].tolist()==job['score']
    assert all(np.isfinite(value).all() for value in arrays.values())
    model=study.model_for(job);state=torch.load(ROOT/comp['checkpoint'],map_location='cpu',weights_only=True)
    assert all(torch.isfinite(value).all() for value in state.values());model.load_state_dict(state,strict=True)
    budget=next(row for row in core.read_json(s/'preflight_validation.json')['source_budgets'] if row['arm']==job['arm'])
    assert sum(p.numel() for p in model.parameters())==budget['parameters']
    if job['arm']!='dino_full':
        bn_names=[name for name,module in model.named_modules() if isinstance(module,torch.nn.BatchNorm2d)];assert len(bn_names)==36
        for name in bn_names:
            for key in ['running_mean','running_var','num_batches_tracked']:
                if job['arm']=='resnet_bn_frozen':assert torch.equal(state[name+'.'+key],cnn_initial[name+'.'+key])
            if job['arm']=='resnet_bn_train':assert state[name+'.num_batches_tracked']>cnn_initial[name+'.num_batches_tracked']
        assert any(not torch.equal(state[name+'.weight'],cnn_initial[name+'.weight']) for name in bn_names)
        source={k:v for k,v in state.items() if k not in ['fc.weight','fc.bias']};initial=cnn_initial
    else:source={k:v.detach().cpu() for k,v in model.source_state().items()};initial=vit_initial
    squares=count=0
    for name,value in source.items():squares+=float((value.double()-initial[name].double()).square().sum());count+=value.numel()
    rms=(squares/count)**.5;assert rms>0
    model.eval().requires_grad_(False);before=state_sha(model);assert before==meta['state_sha256']
    differences={}
    with torch.inference_mode():
        for part,key in [('inner','select'),('score','score')]:
            logits=[];ids=[]
            for x,_,_,indices in core.loader(job[key],False,seed=job['seed']):
                with torch.autocast('cuda',dtype=torch.float16):value=model(x.cuda(non_blocking=True))
                logits.append(value.float().cpu().numpy());ids+=indices.tolist()
            difference=float(np.abs(np.concatenate(logits)-arrays[part+'_logits']).max())
            assert difference<2e-5 and ids==job[key];differences[part]=difference
    assert state_sha(model)==before;del model,state,source;torch.cuda.empty_cache()
    temp=core.read_json(out/'temperature.json');beta=temp['beta']
    assert temp['fit_indices']==job['select'] and temp['signature']==core.signature(job)
    z=arrays['inner_logits'].astype(np.float64);y=np.array([records[i]['label'] for i in job['select']])
    assert .05<=beta<=20 and abs(temp['temperature']-1/beta)<1e-12
    def loss(b):return float(np.mean(logsumexp(b*z,axis=1)-b*z[np.arange(len(y)),y]))
    def gradient(b):return float(np.mean((softmax(b*z)*z).sum(1)-z[np.arange(len(y)),y]))
    if gradient(.05)>=0:optimal=.05
    elif gradient(20.)<=0:optimal=20.
    else:optimal=brentq(gradient,.05,20.,xtol=1e-12)
    assert loss(beta)-loss(optimal)<1e-8 and loss(beta)<=loss(1.)+1e-8
    assert abs(temp['inner_nll_before']-loss(1.))<1e-10 and abs(temp['inner_nll_after']-loss(beta))<1e-10
    outer=arrays['score_logits'].astype(np.float64);p=softmax(beta*outer);rr=study.arch.read_csv(out/'outer_predictions.csv')
    assert [int(row['index']) for row in rr]==job['score']
    saved=np.array([[float(row[f'p{k}']) for k in range(3)] for row in rr])
    assert np.max(np.abs(p-saved))<1e-10 and np.array_equal(p.argmax(1),outer.argmax(1))
    raw=study.arch.read_csv(out/'raw_oof_predictions.csv')
    assert [int(row['index']) for row in raw]==job['score'] and np.array_equal(p.argmax(1),[int(row['pred']) for row in raw])
    inner=study.arch.read_csv(out/'inner_predictions.csv')
    assert [int(row['index']) for row in inner]==job['select']
    inner_f1=f1_score(y,z.argmax(1),average='macro',labels=[0,1,2],zero_division=0)
    assert abs(inner_f1-comp['best_select_f1'])<1e-12
    yy=np.array([records[i]['label'] for i in job['score']]);outer_f1=f1_score(yy,p.argmax(1),average='macro',labels=[0,1,2],zero_division=0)
    assert abs(outer_f1-outer_comp['macro_f1'])<1e-12
    if job['arm']!='dino_full':assert abs(outer_f1-comp['outer_metrics']['macro_f1'])<1e-12
    for row in rr:
        m=records[int(row['index'])];assert m['label']==int(row['label']) and m['group_id']==row['group_id'] and m['sample_id']==row['sample_id']
    unit_predictions[job['arm'],job['seed'],job['fold']]=dict(indices=job['score'],p=p)
    assert (ROOT/comp['checkpoint']).owner()=='login' and all(f.owner()=='login' for f in out.rglob('*') if f.is_file())
    audit.append(dict(job=job['id'],checkpoint_sha256=comp['checkpoint_sha256'],cache_sha256=meta['sha256'],source_rms_drift=rms,beta=beta,
                      independent_optimal_beta=optimal,independent_loss_gap=loss(beta)-loss(optimal),prediction_max_differences=differences))
    curves.append(dict(job=job['id'],arm=job['arm'],seed=job['seed'],fold=job['fold'],best_inner_f1=inner_f1,
                       last_inner_f1=float(log[-1]['val_macro_f1']),last_train_accuracy=float(log[-1]['train_acc']),outer_f1=outer_f1,best_epoch=comp['best_epoch']))
    print('audited',job['id'],flush=True)

schemes={'resnet_bn_train':['resnet_bn_train'],'resnet_bn_frozen':['resnet_bn_frozen'],'dino_full':['dino_full'],
         'mix_train_dino':['resnet_bn_train','dino_full'],'mix_frozen_dino':['resnet_bn_frozen','dino_full'],
         'mix_resnets':['resnet_bn_train','resnet_bn_frozen']}
groups=sorted({m['group_id'] for m in records if m['split']=='train'});assert len(groups)==185
results=core.read_json(s/'round13_results.json');matrices={};merged_preds={}
for scheme,arms in schemes.items():
    for seed in cfg['seeds']:
        prob=np.zeros((2386,3));coverage=np.zeros(2386,dtype=np.int64)
        for fold in range(5):
            sources=[unit_predictions[arm,seed,fold] for arm in arms]
            ids=sources[0]['indices'];assert all(part['indices']==ids for part in sources)
            prob[ids]=sum(part['p'] for part in sources)/len(sources);coverage[ids]+=1
        assert (coverage==1).all()
        rr=study.arch.read_csv(s/'oof'/f'{scheme}_s{seed}.csv')
        assert [int(row['index']) for row in rr]==list(range(2386))
        saved=np.array([[float(row[f'p{k}']) for k in range(3)] for row in rr])
        assert np.max(np.abs(prob-saved))<1e-10
        y=np.array([m['label'] for m in records[:2386]]);pred=prob.argmax(1)
        assert np.array_equal(pred,[int(row['pred']) for row in rr])
        r=next(row for row in results if row['scheme']==scheme and row['seed']==seed)
        score=f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0);cm=confusion_matrix(y,pred,labels=[0,1,2])
        assert abs(score-r['macro_f1'])<1e-12 and cm.tolist()==r['confusion_matrix']
        assert abs(float(-np.log(np.maximum(prob[np.arange(2386),y],1e-300)).mean())-r['nll'])<1e-10
        assert abs(np.square(prob-np.eye(3)[y]).sum(1).mean()-r['brier'])<1e-10
        m=np.zeros((185,3,3));np.add.at(m,([groups.index(row['group_id']) for row in rr],y,pred),1)
        assert abs(np.mean(np.trace(m,axis1=-2,axis2=-1)/m.sum((1,2)))-r['patient_mean_accuracy'])<1e-12
        assert abs(cm[1,1]/cm[1].sum()-r['malignant_recall'])<1e-12
        matrices[scheme,seed]=m;merged_preds[scheme,seed]=pred
weights=np.random.default_rng(20261024).multinomial(185,np.ones(185)/185,size=20000)
def f1(m):
    den=m.sum(-1)+m.sum(-2)
    return np.mean(np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),den,out=np.zeros_like(den),where=den!=0),axis=-1)
comparisons=[('mix_train_dino','resnet_bn_train'),('mix_train_dino','dino_full'),('mix_train_dino','mix_resnets')]
stored=core.read_json(s/'paired_comparisons.json');gates=[]
for a,b in comparisons:
    ds=[];boot=[]
    for seed in cfg['seeds']:
        ma,mb=matrices[a,seed],matrices[b,seed]
        ds.append(float(f1(ma.sum(0))-f1(mb.sum(0))))
        boot.append(f1(np.einsum('bg,gij->bij',weights,ma))-f1(np.einsum('bg,gij->bij',weights,mb)))
    distribution=np.mean(boot,axis=0);family=np.percentile(distribution,[2.5/3,100-2.5/3]);r=stored['comparisons'][a+'_vs_'+b]
    assert np.allclose(ds,r['seed_deltas'],atol=1e-12) and np.allclose(np.percentile(distribution,[2.5,97.5]),r['patient95'],atol=1e-12)
    assert np.allclose(family,r['family_interval'],atol=1e-12)
    gate=min(ds)>0 and np.mean(ds)>=.01 and family[0]>0;assert bool(gate)==r['strong_pair'];gates.append(gate)
assert bool(all(gates[:3]))==stored['strong_primary_followup']
y=np.array([m['label'] for m in records[:2386]])
for row in core.read_json(s/'error_overlap.json'):
    pa,pb=merged_preds[row['a'],row['seed']],merged_preds[row['b'],row['seed']]
    ea,eb=pa!=y,pb!=y
    assert row['both_wrong']==int((ea&eb).sum()) and row['only_a_wrong']==int((ea&~eb).sum()) and row['only_b_wrong']==int((~ea&eb).sum())
    assert abs(row['prediction_disagreement']-np.mean(pa!=pb))<1e-12
assert len(matrices)==18 and all(f.owner()=='login' for f in s.rglob('*') if f.is_file())
core.save_json(s/'training_curve_audit.json',curves)
core.save_json(s/'completion_audit.json',dict(passed=True,timestamp=core.timestamp(),new_neural_trainings=45,source_units=45,temperature_fits=45,merged_oof_files=18,
    checks=['45 new signed checkpoints,60ep,inner selection,finite weights and frozen-BN/source-drift checks',
            'source freeze/eval/state hash,all inner/outer logits independently reproduced','signed finite caches and exact effective indices',
            '45 inner-only temperature fit indices,NLL and independent monotone-gradient convex optimum/bound',
            'positive-temperature class decisions identical to source pipelines','all fixed samefold probability averages reproduced',
            '18 complete2386 OOF/F1/confusion/recall/patient accuracy/NLL/Brier','185-patient five family comparisons and complete primary gate',
            'error overlap and login ownership'],units=audit,comparisons=stored))
print('PASS independent round13 audit',flush=True)
