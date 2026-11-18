"""Independent R12 audit: source inference, scalar convex optima, fixed mixes."""
from pathlib import Path
import sys
import collections
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.optimize import brentq
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_fusion_study as study
from pretrained_representations import state_sha
core=study.core;cfg=study.verify();s=study.STUDY
records=core.read_json(s/'manifest.json');jobs=core.read_json(s/'final_jobs.json')
original=core.read_json(study.DATA/'oof_jobs.json');assert len(jobs)==45
assert core.digest(core.CACHE/'resized_uint8.npy')==core.read_json(core.CACHE/'meta.json')['cache_sha256']

def softmax(z):
    z=np.asarray(z,dtype=np.float64);shift=z-z.max(1,keepdims=True);value=np.exp(shift)
    return value/value.sum(1,keepdims=True)

unit_predictions={};audit=[]
for job in jobs:
    assert study.done(job)
    old=next(j for j in original if j['seed']==job['seed'] and j['fold']==job['fold'])
    assert job['fit']==old['train'] and job['inner']==old['select'] and job['score']==old['score']
    assert all(0<=i<2386 for key in ['fit','inner','score'] for i in job[key])
    core.validate_split(records,job['fit'],job['inner'],job['score'])
    source=ROOT/job['source_dir'];sourcecomp=core.read_json(source/'runs'/job['source_job']['id']/'complete.json')
    assert sourcecomp['signature']==job['source_signature']==study.prior.source_signature(job['source_job'],source)
    assert core.digest(ROOT/job['checkpoint'])==sourcecomp['checkpoint_sha256']==job['checkpoint_sha256']
    rawpath=source/'runs'/job['source_job']['id']/'oof_predictions.csv';assert core.digest(rawpath)==job['source_prediction_sha256']
    meta=core.read_json(study.CACHE/(job['id']+'.json'));path=study.CACHE/(job['id']+'.npz')
    assert meta['signature']==core.signature(job) and core.digest(path)==meta['sha256']
    assert meta['effective_indices']==job['inner']+job['score'] and meta['state_unchanged']
    with np.load(path,allow_pickle=False) as data:arrays={key:data[key] for key in data.files}
    assert arrays['inner_indices'].tolist()==job['inner'] and arrays['score_indices'].tolist()==job['score']
    assert arrays['inner_logits'].shape==(len(job['inner']),3) and arrays['score_logits'].shape==(len(job['score']),3)
    assert all(np.isfinite(value).all() for value in arrays.values())
    model,before=study.model_for(job);assert before==meta['source_state_sha256']
    assert all(not p.requires_grad for p in model.parameters()) and all(not m.training for m in model.modules())
    differences={}
    with torch.inference_mode():
        for part in ['inner','score']:
            logits=[];ids=[]
            for x,_,_,indices in core.loader(job[part],False,seed=job['seed']):
                with torch.autocast('cuda',dtype=torch.float16):value=model(x.cuda(non_blocking=True))
                logits.append(value.float().cpu().numpy());ids+=indices.tolist()
            difference=float(np.abs(np.concatenate(logits)-arrays[part+'_logits']).max())
            assert difference<2e-5 and ids==job[part];differences[part]=difference
    assert state_sha(model)==before;del model;torch.cuda.empty_cache()
    out=s/'runs'/job['id'];temp=core.read_json(out/'temperature.json');comp=core.read_json(out/'complete.json')
    assert temp['fit_indices']==job['inner'] and temp['signature']==core.signature(job)
    assert temp['source_checkpoint_sha256']==job['checkpoint_sha256']
    z=arrays['inner_logits'].astype(np.float64);y=np.array([records[i]['label'] for i in job['inner']]);beta=temp['beta']
    assert .05<=beta<=20 and abs(temp['temperature']-1/beta)<1e-12
    def loss(b):return float(np.mean(logsumexp(b*z,axis=1)-b*z[np.arange(len(y)),y]))
    def gradient(b):return float(np.mean((softmax(b*z)*z).sum(1)-z[np.arange(len(y)),y]))
    if gradient(.05)>=0:optimal=.05
    elif gradient(20.)<=0:optimal=20.
    else:optimal=brentq(gradient,.05,20.,xtol=1e-12)
    assert loss(beta)-loss(optimal)<1e-8 and loss(beta)<=loss(1.)+1e-8
    assert abs(temp['inner_nll_before']-loss(1.))<1e-10 and abs(temp['inner_nll_after']-loss(beta))<1e-10
    assert abs(temp['beta_gradient']-gradient(beta))<1e-10
    outer=arrays['score_logits'].astype(np.float64);p=softmax(beta*outer);rr=study.arch.read_csv(out/'outer_predictions.csv')
    assert [int(row['index']) for row in rr]==job['score']
    saved=np.array([[float(row[f'p{k}']) for k in range(3)] for row in rr])
    assert np.max(np.abs(p-saved))<1e-10 and np.array_equal(p.argmax(1),outer.argmax(1))
    for row in rr:
        m=records[int(row['index'])]
        assert m['label']==int(row['label']) and m['group_id']==row['group_id'] and m['sample_id']==row['sample_id']
    raw=sorted(study.arch.read_csv(rawpath),key=lambda row:int(row['index']))
    order=np.argsort(job['score']);assert np.array_equal(p.argmax(1)[order],[int(row['pred']) for row in raw])
    unit_predictions[job['arm'],job['seed'],job['fold']]=dict(indices=job['score'],p=p)
    assert core.digest(out/'temperature.json')==comp['temperature_sha256']
    assert core.digest(out/'outer_predictions.csv')==comp['prediction_sha256']
    assert path.owner()=='login' and all(f.owner()=='login' for f in out.rglob('*') if f.is_file())
    audit.append(dict(job=job['id'],source_checkpoint_sha256=job['checkpoint_sha256'],cache_sha256=meta['sha256'],beta=beta,
                      independent_optimal_beta=optimal,independent_loss_gap=loss(beta)-loss(optimal),prediction_max_differences=differences))
    print('audited',job['id'],flush=True)

schemes={'resnet_bn_train':['resnet_bn_train'],'resnet_bn_frozen':['resnet_bn_frozen'],'dino_full':['dino_full'],
         'mix_train_dino':['resnet_bn_train','dino_full'],'mix_frozen_dino':['resnet_bn_frozen','dino_full'],
         'mix_resnets':['resnet_bn_train','resnet_bn_frozen']}
groups=sorted({m['group_id'] for m in records if m['split']=='train'});assert len(groups)==185
results=core.read_json(s/'round12_results.json');matrices={};merged_preds={}
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
        if len(arms)==1:
            old=next(row for row in core.read_json(study.PREVIOUS/'round11_results.json') if row['scheme']==scheme and row['seed']==seed)
            assert abs(old['macro_f1']-score)<1e-12
weights=np.random.default_rng(20261024).multinomial(185,np.ones(185)/185,size=20000)
def f1(m):
    den=m.sum(-1)+m.sum(-2)
    return np.mean(np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),den,out=np.zeros_like(den),where=den!=0),axis=-1)
comparisons=[('mix_frozen_dino','dino_full'),('mix_frozen_dino','resnet_bn_frozen'),('mix_frozen_dino','mix_resnets'),
             ('mix_train_dino','dino_full'),('mix_train_dino','resnet_bn_train')]
stored=core.read_json(s/'paired_comparisons.json');gates=[]
for a,b in comparisons:
    ds=[];boot=[]
    for seed in cfg['seeds']:
        ma,mb=matrices[a,seed],matrices[b,seed]
        ds.append(float(f1(ma.sum(0))-f1(mb.sum(0))))
        boot.append(f1(np.einsum('bg,gij->bij',weights,ma))-f1(np.einsum('bg,gij->bij',weights,mb)))
    distribution=np.mean(boot,axis=0);family=np.percentile(distribution,[.5,99.5]);r=stored['comparisons'][a+'_vs_'+b]
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
core.save_json(s/'completion_audit.json',dict(passed=True,timestamp=core.timestamp(),new_neural_trainings=0,source_units=45,temperature_fits=45,merged_oof_files=18,
    checks=['45 source signatures/config/checkpoints/raw probabilities,exact original patient triples and onlytrain scope',
            'source freeze/eval/state hash,all inner/outer logits independently reproduced','signed finite caches and exact effective indices',
            '45 inner-only temperature fit indices,NLL and independent monotone-gradient convex optimum/bound',
            'positive-temperature class decisions identical to source pipelines','all fixed samefold probability averages reproduced',
            '18 complete2386 OOF/F1/confusion/recall/patient accuracy/NLL/Brier','185-patient five family comparisons and complete primary gate',
            'error overlap and login ownership'],units=audit,comparisons=stored))
print('PASS independent round12 audit',flush=True)
