#!/usr/bin/env python3
"""Round12: fixed CNN/ViT probability complementarity; no neural training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_dino_oof_study as prior
from prediction_fusion import SCHEMES,COMPARISONS,fit_temperature,probabilities,mixture,probability_scores,nll,beta_gradient
core=prior.core;arch=prior.arch
DATA=prior.DATA;BN=prior.BN;PREVIOUS=prior.STUDY
STUDY=ROOT/'experiments/fusion_study';CACHE=ROOT/'.cache_fusion_study'
SOURCE_PIPELINE=prior.sources()
arch.STUDY=STUDY;arch.__file__=__file__;core.STUDY=STUDY


def save_json(path,value):
    if Path(path)==STUDY/'status.json' and value.get('phase')=='round1_training':
        value=dict(value,phase='round12_fixed_fusion',round=12,new_neural_trainings=0,temperature_fits=45)
    prior.transfer.SAVE(path,value)


core.save_json=save_json
arch.commit=lambda message:prior.transfer.COMMIT(message.replace('development-only architecture run','frozen-source temperature unit'))
arch.baseline_predictions=lambda:None


def sources():
    import scipy,inspect
    from scipy.optimize import minimize_scalar
    return dict(driver=core.digest(__file__),helper=core.digest(ROOT/'experiments/prediction_fusion.py'),
                protocol=core.digest(ROOT/'experiments/ROUND12_PROTOCOL.md'),source_pipeline=SOURCE_PIPELINE,scipy=scipy.__version__,
                scipy_minimizer_sha256=core.digest(inspect.getsourcefile(minimize_scalar)))


def verify():
    cfg=core.read_json(STUDY/'config.json');assert cfg['architecture_sources']==sources()
    for path,key in [(STUDY/'manifest.json','manifest_sha256'),(STUDY/'final_jobs.json','jobs_sha256'),
                     (core.CACHE/'meta.json','cache_meta_sha256'),(DATA/'oof_jobs.json','source_folds_sha256'),
                     (PREVIOUS/'completion_audit.json','source_audit_sha256')]:assert core.digest(path)==cfg[key]
    for relative,sha in cfg['source_configs'].items():assert core.digest(ROOT/relative)==sha
    return cfg


arch.verify=verify


def prepare():
    STUDY.mkdir(parents=True,exist_ok=True);CACHE.mkdir(exist_ok=True)
    if (STUDY/'config.json').exists():verify();return
    assert core.read_json(PREVIOUS/'completion_audit.json')['passed']
    (STUDY/'manifest.json').write_bytes((PREVIOUS/'manifest.json').read_bytes())
    units=[];references=core.read_json(PREVIOUS/'reference_index.json');dinojobs=core.read_json(PREVIOUS/'final_jobs.json')
    for entry in references:
        units.append(dict(arm=entry['arm'],source_dir=entry['source_dir'],source_job=entry['source_job']))
    units += [dict(arm='dino_full',source_dir=str(PREVIOUS.relative_to(ROOT)),source_job=job) for job in dinojobs]
    jobs=[];configs={}
    for unit in units:
        source=ROOT/unit['source_dir'];job=unit['source_job'];comp=core.read_json(source/'runs'/job['id']/'complete.json')
        assert comp['signature']==prior.source_signature(job,source)
        assert core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']
        configs[str((source/'config.json').relative_to(ROOT))]=core.digest(source/'config.json')
        jobs.append(dict(id=f'r12_{unit["arm"]}_s{job["seed"]}_f{job["fold"]}',arm=unit['arm'],seed=job['seed'],fold=job['fold'],
                         source_dir=unit['source_dir'],source_job=job,source_signature=comp['signature'],checkpoint=comp['checkpoint'],
                         checkpoint_sha256=comp['checkpoint_sha256'],source_prediction_sha256=core.digest(source/'runs'/job['id']/'oof_predictions.csv'),
                         inner=job['select'],score=job['score'],fit=job['train'],architecture_sources=sources()))
    assert len(jobs)==45
    core.save_json(STUDY/'final_jobs.json',jobs)
    cfg=dict(round=12,created_at=core.timestamp(),architecture_sources=sources(),seeds=[42,43,44],schemes=list(SCHEMES),
             primary='mix_frozen_dino',new_neural_trainings=0,temperature_fits=45,source_units=45,
             manifest_sha256=core.digest(STUDY/'manifest.json'),jobs_sha256=core.digest(STUDY/'final_jobs.json'),
             source_configs=configs,source_folds_sha256=core.digest(DATA/'oof_jobs.json'),source_audit_sha256=core.digest(PREVIOUS/'completion_audit.json'),
             cache_meta_sha256=core.digest(core.CACHE/'meta.json'),effective_indices=list(range(2386)),beta_bounds=[.05,20],
             calibration='Only inner original-label NLL, bounded convex beta, no outer-label access; inner also selected checkpoint.',
             mixture_weights=[.5,.5],max_parallel=2,cpu_threads=2,bootstrap=dict(groups=185,draws=20000,seed=20261024,family_percentiles=[.5,99.5]),
             policy='Six fixed probability pipelines; original train only, no grades/relabeling/fixed val/test; not new architecture or compute matching.')
    core.save_json(STUDY/'config.json',cfg)
    core.save_json(STUDY/'status.json',dict(phase='prepared',round=12,total=45,new_neural_trainings=0,updated_at=core.timestamp()))
    arch.commit('research: freeze equal-weight CNN-DINO complementarity diagnosis and 45 source units')


def model_for(job):
    import torch
    from pretrained_representations import state_sha
    if job['arm']=='dino_full':model=prior.transfer.model_for(job['seed'],'full_finetune')
    else:model=arch.ORIGINAL_MODEL(job['seed'])
    model.load_state_dict(torch.load(ROOT/job['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False)
    return model,state_sha(model)


def extract(job):
    import torch
    from pretrained_representations import state_sha
    path=CACHE/(job['id']+'.npz');meta=CACHE/(job['id']+'.json');signature=core.signature(job)
    raw_path=ROOT/job['source_dir']/'runs'/job['source_job']['id']/'oof_predictions.csv'
    assert core.digest(ROOT/job['checkpoint'])==job['checkpoint_sha256']
    assert core.digest(raw_path)==job['source_prediction_sha256']
    if meta.exists():
        value=core.read_json(meta)
        assert value['signature']==signature and core.digest(path)==value['sha256'] and value['source_checkpoint_sha256']==job['checkpoint_sha256']
        with np.load(path,allow_pickle=False) as data:return {key:data[key] for key in data.files},value
    assert core.digest(ROOT/job['checkpoint'])==job['checkpoint_sha256']
    model,before=model_for(job);arrays={};start=time.time()
    with torch.inference_mode():
        for part in ['inner','score']:
            chunks=[];indices=[]
            for x,_,_,idx in core.loader(job[part],False,seed=job['seed']):
                with torch.autocast('cuda',dtype=torch.float16):logits=model(x.cuda(non_blocking=True))
                chunks.append(logits.float().cpu().numpy());indices+=idx.tolist()
            assert indices==job[part]
            arrays[part+'_logits']=np.concatenate(chunks);arrays[part+'_indices']=np.array(indices)
    assert all(not p.requires_grad for p in model.parameters()) and all(not m.training for m in model.modules())
    assert state_sha(model)==before
    raw=sorted(arch.read_csv(raw_path),key=lambda row:int(row['index']))
    original=np.array([[float(row[f'p{k}']) for k in range(3)] for row in raw])
    order=np.argsort(arrays['score_indices'])
    with torch.no_grad():reproduced=torch.from_numpy(arrays['score_logits']).softmax(-1).numpy()[order]
    difference=float(np.abs(original-reproduced).max())
    assert difference<2e-6 and np.array_equal(original.argmax(1),reproduced.argmax(1))
    assert [int(row['index']) for row in raw]==arrays['score_indices'][order].tolist()
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:np.savez(f,**arrays)
    temporary.replace(path)
    value=dict(signature=signature,sha256=core.digest(path),source_checkpoint_sha256=job['checkpoint_sha256'],
               source_state_sha256=before,state_unchanged=True,effective_indices=job['inner']+job['score'],
               raw_probability_max_absolute_diff=difference,seconds=time.time()-start)
    core.save_json(meta,value);del model;torch.cuda.empty_cache()
    return arrays,value


def worker(job_id):
    verify();job=next(job for job in core.read_json(STUDY/'final_jobs.json') if job['id']==job_id)
    if done(job):return
    arrays,meta=extract(job);records=core.read_json(STUDY/'manifest.json')
    labels=np.array([records[i]['label'] for i in job['inner']])
    temperature=fit_temperature(arrays['inner_logits'],labels)
    temperature.update(fit_indices=job['inner'],source_checkpoint_sha256=job['checkpoint_sha256'],signature=core.signature(job))
    out=STUDY/'runs'/job_id;core.save_json(out/'temperature.json',temperature)
    p=probabilities(arrays['score_logits'],temperature['beta']);y=np.array([records[i]['label'] for i in job['score']])
    assert np.array_equal(p.argmax(1),arrays['score_logits'].argmax(1))
    core.save_csv(out/'outer_predictions.csv',core.prediction_rows(p,y,job['score']))
    core.save_json(out/'complete.json',dict(signature=core.signature(job),job_id=job_id,arm=job['arm'],seed=job['seed'],fold=job['fold'],
        n=len(y),new_neural_trainings=0,cache_sha256=meta['sha256'],temperature_sha256=core.digest(out/'temperature.json'),
        prediction_sha256=core.digest(out/'outer_predictions.csv'),source_checkpoint_sha256=job['checkpoint_sha256'],
        completed_at=core.timestamp(),**core.metrics(p,y),**probability_scores(p,y)))


def done(job):
    path=STUDY/'runs'/job['id']/'complete.json'
    if not path.exists():return False
    comp=core.read_json(path)
    assert comp['signature']==core.signature(job) and comp['n']==len(job['score'])
    for name,key in [('temperature.json','temperature_sha256'),('outer_predictions.csv','prediction_sha256')]:assert core.digest(path.parent/name)==comp[key]
    assert core.digest(CACHE/(job['id']+'.npz'))==comp['cache_sha256']
    assert core.digest(ROOT/job['checkpoint'])==comp['source_checkpoint_sha256']==job['checkpoint_sha256']
    assert core.digest(ROOT/job['source_dir']/'runs'/job['source_job']['id']/'oof_predictions.csv')==job['source_prediction_sha256']
    return True


arch.done=done


def check():
    import torch
    cfg=verify();records=core.read_json(STUDY/'manifest.json');jobs=core.read_json(STUDY/'final_jobs.json')
    original=core.read_json(DATA/'oof_jobs.json')
    assert core.digest(core.CACHE/'resized_uint8.npy')==core.read_json(core.CACHE/'meta.json')['cache_sha256']
    coverage={}
    for job in jobs:
        old=next(j for j in original if j['seed']==job['seed'] and j['fold']==job['fold'])
        assert job['fit']==old['train'] and job['inner']==old['select'] and job['score']==old['score']
        assert all(0<=i<2386 for key in ['fit','inner','score'] for i in job[key])
        core.validate_split(records,job['fit'],job['inner'],job['score'])
        assert core.digest(ROOT/job['checkpoint'])==job['checkpoint_sha256']
        assert job['source_signature']==prior.source_signature(job['source_job'],ROOT/job['source_dir'])
        coverage.setdefault((job['arm'],job['seed']),[]).extend(job['score'])
    assert len(coverage)==9 and all(sorted(ids)==list(range(2386)) for ids in coverage.values())
    z=np.array([[3.,1.,-2.],[-1.,.7,2.],[.2,.1,-.4],[2.,2.5,-1.]])
    y=np.array([0,1,2,1]);beta=.71;h=1e-5
    assert abs((nll(beta+h,z,y)-nll(beta-h,z,y))/(2*h)-beta_gradient(beta,z,y))<1e-8
    fitted=fit_temperature(z,y)
    assert fitted['inner_nll_after']<=min(nll(b,z,y) for b in np.linspace(.05,20,2000))+1e-8
    for b in [.05,.4,1.,20]:assert np.array_equal(probabilities(z,b).argmax(1),z.argmax(1))
    p,q=probabilities(z,1),probabilities(z,.5)
    assert np.allclose(mixture([p,q]),(p+q)/2) and np.allclose(mixture([p]),p)
    checks=[]
    # One source of each architecture/norm level, caches reused by workers.
    for arm in ['resnet_bn_train','resnet_bn_frozen','dino_full']:
        job=next(j for j in jobs if j['arm']==arm and j['seed']==42 and j['fold']==0)
        arrays,meta=extract(job);assert meta['state_unchanged']
        assert set(meta['effective_indices'])==set(job['inner']+job['score'])
        value=fit_temperature(arrays['inner_logits'],np.array([records[i]['label'] for i in job['inner']]))
        assert value['inner_nll_after']<=value['inner_nll_before']+1e-8
        checks.append(dict(arm=arm,source_state_sha256=meta['source_state_sha256'],raw_probability_max_absolute_diff=meta['raw_probability_max_absolute_diff']))
    core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),new_neural_trainings=0,temperature_fits=45,
        cases=checks,checks=['45 signed checkpoints,exact original15folds and patient isolation','only original train indices,source coverage2386 per arm/seed',
            'full cache SHA and source/runtime hashes','independent calibration finite-difference gradient and convex-grid objective',
            'positive-temperature decisions invariant and arithmetic mean reference','three source levels eval/frozen/state unchanged/raw probabilities reproduced',
            'only inner logits and labels passed to fitting; outer labels only used for scoring']))
    arch.commit('test: validate nested temperature algebra, fixed mixtures and frozen-source reproduction')
    print('PASS round12 inference/calibration preflight',flush=True)


def summarize():
    cfg=verify();jobs=core.read_json(STUDY/'final_jobs.json');records=core.read_json(STUDY/'manifest.json')
    predictions={};summary=[];folds=[];overlap=[]
    for scheme,arms in SCHEMES.items():
        for seed in cfg['seeds']:
            merged=[]
            for fold in range(5):
                parts=[];indices=None
                for arm in arms:
                    job=next(j for j in jobs if j['arm']==arm and j['seed']==seed and j['fold']==fold);assert done(job)
                    rr=arch.read_csv(STUDY/'runs'/job['id']/'outer_predictions.csv')
                    ids=[int(row['index']) for row in rr]
                    if indices is None:indices=ids
                    assert indices==ids
                    parts.append(np.array([[float(row[f'p{k}']) for k in range(3)] for row in rr]))
                p=mixture(parts);y=np.array([records[i]['label'] for i in indices])
                merged+=core.prediction_rows(p,y,indices)
                folds.append(dict(scheme=scheme,seed=seed,fold=fold,n=len(y),**core.metrics(p,y),**probability_scores(p,y)))
            merged=sorted(merged,key=lambda row:int(row['index']));assert [int(row['index']) for row in merged]==list(range(2386))
            core.save_csv(STUDY/'oof'/f'{scheme}_s{seed}.csv',merged);predictions[scheme,seed]=merged
            p=np.array([[row[f'p{k}'] for k in range(3)] for row in merged]);y=np.array([row['label'] for row in merged])
            metric=core.metrics(p,y);cm=np.array(metric['confusion_matrix']);groups=sorted({row['group_id'] for row in merged})
            patient=np.mean([np.mean([row['pred']==row['label'] for row in merged if row['group_id']==g]) for g in groups])
            summary.append(dict(scheme=scheme,seed=seed,n=2386,malignant_recall=float(cm[1,1]/cm[1].sum()),patient_mean_accuracy=float(patient),
                                **metric,**probability_scores(p,y)))
    for seed in cfg['seeds']:
        for a,b in [('resnet_bn_train','dino_full'),('resnet_bn_frozen','dino_full'),('resnet_bn_train','resnet_bn_frozen')]:
            ra,rb=predictions[a,seed],predictions[b,seed]
            ea=np.array([r['pred']!=r['label'] for r in ra]);eb=np.array([r['pred']!=r['label'] for r in rb])
            overlap.append(dict(seed=seed,a=a,b=b,both_wrong=int((ea&eb).sum()),only_a_wrong=int((ea&~eb).sum()),only_b_wrong=int((~ea&eb).sum()),
                                prediction_disagreement=float(np.mean([x['pred']!=y['pred'] for x,y in zip(ra,rb)]))))
    weights=np.random.default_rng(20261024).multinomial(len(groups),np.ones(len(groups))/len(groups),size=20000)
    def matrix(rr):
        m=np.zeros((len(groups),3,3));np.add.at(m,([groups.index(row['group_id']) for row in rr],[int(row['label']) for row in rr],[int(row['pred']) for row in rr]),1);return m
    def f1(m):
        den=m.sum(-1)+m.sum(-2);return np.mean(np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),den,out=np.zeros_like(den),where=den!=0),axis=-1)
    comparisons={}
    for a,b in COMPARISONS:
        ds=[];boot=[]
        for seed in cfg['seeds']:
            ma,mb=matrix(predictions[a,seed]),matrix(predictions[b,seed])
            ds.append(float(f1(ma.sum(0))-f1(mb.sum(0))))
            boot.append(f1(np.einsum('bg,gij->bij',weights,ma))-f1(np.einsum('bg,gij->bij',weights,mb)))
        distribution=np.mean(boot,axis=0);family=np.percentile(distribution,[.5,99.5]).tolist()
        comparisons[a+'_vs_'+b]=dict(seed_deltas=ds,mean_delta=float(np.mean(ds)),patient95=np.percentile(distribution,[2.5,97.5]).tolist(),family_interval=family,
                                   strong_pair=bool(min(ds)>0 and np.mean(ds)>=.01 and family[0]>0))
    strong=all(comparisons[a+'_vs_'+b]['strong_pair'] for a,b in COMPARISONS[:3])
    core.save_json(STUDY/'round12_results.json',summary);core.save_json(STUDY/'fold_results.json',folds);core.save_json(STUDY/'error_overlap.json',overlap)
    core.save_json(STUDY/'paired_comparisons.json',dict(comparisons=comparisons,strong_primary_followup=strong))
    lines=['# Round12: fixed CNN/ViT complementarity','',
           'Zero new neural trainings;45 reused source models,45 inner-only temperatures,6 fixed pipelines. No fixed val/test or relabeling.',
           'Mature ensemble/calibration diagnostic, not new architecture or budget matching.','',
           '| Pipeline | Mean pooled OOF F1 | Malignant recall | NLL | Brier |','|---|---:|---:|---:|---:|']
    for arm in SCHEMES:
        rr=[r for r in summary if r['scheme']==arm]
        lines.append(f'| {arm} | {np.mean([r["macro_f1"] for r in rr]):.6f} | {np.mean([r["malignant_recall"] for r in rr]):.6f} | {np.mean([r["nll"] for r in rr]):.6f} | {np.mean([r["brier"] for r in rr]):.6f} |')
    lines+=['',f'Complete preregistered primary follow-up gate: {strong}.','Intervals condition on repeated-development cross-fitted sources; not external confirmation or clinical confidence.']
    (STUDY/'ROUND12_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    arch.commit('results: summarize fixed cross-family probability fusion and paired patient uncertainty')
    core.save_json(STUDY/'status.json',dict(phase='round12_complete_needs_analysis',round=12,total=45,completed=45,new_neural_trainings=0,updated_at=core.timestamp()))


arch.summarize=summarize
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','worker','run']);parser.add_argument('--job');args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='check':check()
    elif args.command=='worker':worker(args.job)
    else:arch.run()
