#!/usr/bin/env python3
"""R13: preregistered replication of a secondary fixed-fusion finding."""
import argparse
import collections
import json
from pathlib import Path
import sys
import types
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_fusion_study as previous
from prediction_fusion import SCHEMES,fit_temperature,probabilities,mixture,probability_scores
core=previous.core;arch=previous.arch;transfer=previous.prior.transfer
SOURCE_FUSION=previous.sources()
DATA=previous.DATA;PREVIOUS=previous.STUDY
STUDY=ROOT/'experiments/fusion_confirmation';CACHE=ROOT/'.cache_fusion_confirmation'
ARMS=['resnet_bn_train','resnet_bn_frozen','dino_full'];SEEDS=[45,46,47]
COMPARISONS=[('mix_train_dino','resnet_bn_train'),('mix_train_dino','dino_full'),('mix_train_dino','mix_resnets')]
arch.STUDY=STUDY;arch.__file__=__file__;core.STUDY=STUDY
core.WEIGHTS=ROOT/'weights/fusion_confirmation';transfer.STUDY=STUDY


def save_json(path,value):
    if Path(path)==STUDY/'status.json' and value.get('phase')=='round1_training':
        value=dict(value,phase='round13_fusion_replication',round=13,new_neural_trainings=45)
    transfer.SAVE(path,value)


core.save_json=save_json
arch.commit=lambda message:transfer.COMMIT(message.replace('development-only architecture run','new-split fusion source'))
arch.baseline_predictions=lambda:None


def sources():
    import sklearn
    return dict(driver=core.digest(__file__),protocol=core.digest(ROOT/'experiments/ROUND13_PROTOCOL.md'),
                previous_fusion=SOURCE_FUSION,bn_policy_reference=core.digest(ROOT/'experiments/run_bn_oof_study.py'),sklearn=sklearn.__version__)


def verify():
    cfg=core.read_json(STUDY/'config.json');assert cfg['architecture_sources']==sources()
    for path,key in [(STUDY/'manifest.json','manifest_sha256'),(STUDY/'final_jobs.json','jobs_sha256'),
                     (STUDY/'folds.json','folds_sha256'),(core.CACHE/'meta.json','cache_meta_sha256'),
                     (DATA/'config.json','source_data_config_sha256'),(PREVIOUS/'completion_audit.json','previous_audit_sha256')]:
        assert core.digest(path)==cfg[key]
    assert core.digest(core.PRETRAINED)==cfg['pretrained_sha256']
    assert core.digest(ROOT/'weights/pretrained/dinov2_vits14_pretrain.pth')==cfg['dino_pretrained_sha256']==transfer.DINO_SHA
    return cfg


transfer.verify=verify;arch.verify=verify


def frozen_model(seed):
    import torch
    model=arch.ORIGINAL_MODEL(seed);original_train=model.train
    def training(self,mode=True):
        original_train(mode)
        if mode:
            for module in self.modules():
                if isinstance(module,torch.nn.BatchNorm2d):module.eval()
        return self
    model.train=types.MethodType(training,model)
    return model


def model_for(job):
    if job['arm']=='dino_full':return transfer.model_for(job['seed'],'full_finetune')
    if job['arm']=='resnet_bn_frozen':return frozen_model(job['seed'])
    return arch.ORIGINAL_MODEL(job['seed'])


def prepare():
    from sklearn.model_selection import StratifiedGroupKFold
    STUDY.mkdir(parents=True,exist_ok=True);CACHE.mkdir(exist_ok=True)
    if (STUDY/'config.json').exists():verify();return
    assert core.read_json(PREVIOUS/'completion_audit.json')['passed']
    (STUDY/'manifest.json').write_bytes((DATA/'manifest.json').read_bytes())
    records=core.read_json(STUDY/'manifest.json');indices=np.arange(2386)
    labels=np.array([records[i]['label'] for i in indices]);groups=np.array([records[i]['group_id'] for i in indices])
    folds=[];jobs=[]
    for seed in SEEDS:
        outer=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
        for fold,(rest_pos,score_pos) in enumerate(outer.split(indices,labels,groups)):
            rest,score=indices[rest_pos],indices[score_pos]
            inner=StratifiedGroupKFold(10,shuffle=True,random_state=seed*100+fold)
            selected=None
            for fit_pos,select_pos in inner.split(rest,labels[rest_pos],groups[rest_pos]):
                if all({records[i]['label'] for i in part}=={0,1,2} for part in [rest[fit_pos],rest[select_pos],score]):
                    selected=(rest[fit_pos],rest[select_pos]);break
            assert selected is not None
            fit,select=selected
            core.validate_split(records,fit,select,score)
            shared=dict(seed=seed,fold=fold,train=fit.tolist(),select=select.tolist(),score=score.tolist())
            folds.append(shared)
            for arm in ARMS:
                jobs.append(dict(shared,id=f'r13_{arm}_s{seed}_f{fold}',phase='oof',arm=arm,
                                 variant='full_finetune' if arm=='dino_full' else arm,recipe='full_unweighted',
                                 epochs=60,sample_weights={},architecture_sources=sources()))
    core.save_json(STUDY/'folds.json',folds);core.save_json(STUDY/'final_jobs.json',jobs)
    cfg=dict(round=13,created_at=core.timestamp(),architecture_sources=sources(),seeds=SEEDS,source_arms=ARMS,schemes=list(SCHEMES),
             primary='mix_train_dino',new_neural_trainings=45,temperature_fits=45,active_indices=list(range(2386)),
             manifest_sha256=core.digest(STUDY/'manifest.json'),jobs_sha256=core.digest(STUDY/'final_jobs.json'),folds_sha256=core.digest(STUDY/'folds.json'),
             source_data_config_sha256=core.digest(DATA/'config.json'),previous_audit_sha256=core.digest(PREVIOUS/'completion_audit.json'),
             cache_meta_sha256=core.digest(core.CACHE/'meta.json'),pretrained_sha256=core.digest(core.PRETRAINED),dino_pretrained_sha256=transfer.DINO_SHA,
             cnn_recipe='Unchanged core.run_worker: AdamW1e-4wd.05 cosine0 AMP batch32 60ep, no clipping.',
             dino_recipe='Unchanged R10 train_job: encoder1e-5/head1e-4 wd.05 cosine0 AMP clip1 batch32 60ep.',
             calibration='Unchanged R12 helper, inner-only bounded beta[.05,20], equal .5 probability pairs, no outer tuning.',
             bootstrap=dict(groups=185,draws=20000,seed=20261024,family_percentiles=[2.5/3,100-2.5/3]),
             policy='Replication of R12 secondary, not replacement of its failed primary. Same internal patients,new fixed splits/training, no val/test/grades/relabeling.')
    core.save_json(STUDY/'config.json',cfg)
    core.save_json(STUDY/'status.json',dict(phase='prepared',round=13,total=45,new_neural_trainings=45,updated_at=core.timestamp()))
    arch.commit('research: freeze new45-47 patient folds and 45 source-model replication jobs')


def extract(job,comp):
    import torch
    from pretrained_representations import state_sha
    path=CACHE/(job['id']+'.npz');meta=CACHE/(job['id']+'.json');signature=core.signature(job)
    assert core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']
    if meta.exists():
        value=core.read_json(meta);assert value['signature']==signature and core.digest(path)==value['sha256']
        assert value['checkpoint_sha256']==comp['checkpoint_sha256']
        with np.load(path,allow_pickle=False) as data:return {key:data[key] for key in data.files},value
    model=model_for(job)
    model.load_state_dict(torch.load(ROOT/comp['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False);before=state_sha(model);arrays={}
    with torch.inference_mode():
        for part,key in [('inner','select'),('score','score')]:
            values=[];ids=[]
            for x,_,_,idx in core.loader(job[key],False,seed=job['seed']):
                with torch.autocast('cuda',dtype=torch.float16):z=model(x.cuda(non_blocking=True))
                values.append(z.float().cpu().numpy());ids+=idx.tolist()
            assert ids==job[key]
            arrays[part+'_logits']=np.concatenate(values);arrays[part+'_indices']=np.array(ids)
    assert state_sha(model)==before
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:np.savez(f,**arrays)
    temporary.replace(path)
    value=dict(signature=signature,sha256=core.digest(path),checkpoint_sha256=comp['checkpoint_sha256'],state_sha256=before,
               state_unchanged=True,effective_indices=job['select']+job['score'])
    core.save_json(meta,value);del model;torch.cuda.empty_cache()
    return arrays,value


def worker(job_id):
    import torch
    verify();job=next(j for j in core.read_json(STUDY/'final_jobs.json') if j['id']==job_id)
    if done(job):return
    if not core.complete(job):
        if job['arm']=='dino_full':transfer.train_job(job)
        else:
            core.new_model=frozen_model if job['arm']=='resnet_bn_frozen' else arch.ORIGINAL_MODEL
            core.run_worker(job_id)
    out=STUDY/'runs'/job_id;comp=core.read_json(out/'complete.json');arrays,meta=extract(job,comp)
    records=core.read_json(STUDY/'manifest.json');inner_y=np.array([records[i]['label'] for i in job['select']])
    raw_inner=torch.from_numpy(arrays['inner_logits']).softmax(-1).numpy()
    assert abs(core.metrics(raw_inner,inner_y)['macro_f1']-comp['best_select_f1'])<1e-12
    core.save_csv(out/'inner_predictions.csv',core.prediction_rows(raw_inner,inner_y,job['select']))
    temp=fit_temperature(arrays['inner_logits'],inner_y)
    temp.update(signature=core.signature(job),fit_indices=job['select'],checkpoint_sha256=comp['checkpoint_sha256'])
    core.save_json(out/'temperature.json',temp)
    y=np.array([records[i]['label'] for i in job['score']]);raw=torch.from_numpy(arrays['score_logits']).softmax(-1).numpy()
    p=probabilities(arrays['score_logits'],temp['beta']);assert np.array_equal(p.argmax(1),raw.argmax(1))
    if job['arm']!='dino_full':assert abs(core.metrics(raw,y)['macro_f1']-comp['outer_metrics']['macro_f1'])<1e-12
    core.save_csv(out/'raw_oof_predictions.csv',core.prediction_rows(raw,y,job['score']))
    core.save_csv(out/'outer_predictions.csv',core.prediction_rows(p,y,job['score']))
    hashes={name:core.digest(out/name) for name in ['inner_predictions.csv','raw_oof_predictions.csv','outer_predictions.csv','temperature.json']}
    core.save_json(out/'outer_complete.json',dict(signature=core.signature(job),n=len(y),checkpoint_sha256=comp['checkpoint_sha256'],
        cache_sha256=meta['sha256'],file_hashes=hashes,completed_at=core.timestamp(),**core.metrics(p,y),**probability_scores(p,y)))


def done(job):
    path=STUDY/'runs'/job['id']/'outer_complete.json'
    if not path.exists():return False
    assert core.complete(job)
    result=core.read_json(path)
    assert result['signature']==core.signature(job) and result['n']==len(job['score'])
    for name,sha in result['file_hashes'].items():assert core.digest(path.parent/name)==sha
    assert core.digest(CACHE/(job['id']+'.npz'))==result['cache_sha256']
    return True


arch.done=done


def check():
    import torch
    import torch.nn.functional as F
    from models.optimize.cle_token_adapter import counted_flops
    cfg=verify();records=core.read_json(STUDY/'manifest.json');jobs=core.read_json(STUDY/'final_jobs.json')
    assert len(jobs)==45 and cfg['seeds']==[45,46,47]
    assert core.digest(core.CACHE/'resized_uint8.npy')==core.read_json(core.CACHE/'meta.json')['cache_sha256']
    coverage=collections.defaultdict(list)
    for seed in SEEDS:
        for fold in range(5):
            group=[j for j in jobs if j['seed']==seed and j['fold']==fold];assert len(group)==3
            for key in ['train','select','score']:assert all(j[key]==group[0][key] for j in group)
    for job in jobs:
        core.validate_split(records,job['train'],job['select'],job['score'])
        assert all(0<=i<2386 for key in ['train','select','score'] for i in job[key]) and not job['sample_weights']
        coverage[job['arm'],job['seed']]+=job['score']
    assert len(coverage)==9 and all(sorted(ids)==list(range(2386)) for ids in coverage.values())
    job=jobs[0];x,y,_,_=next(iter(core.loader(job['train'][:32],False,seed=45)));x,y=x.cuda(),y.cuda()
    initial_cnn=arch.ORIGINAL_MODEL(45).eval()
    expected={name:value.detach().cpu().clone() for name,value in initial_cnn.state_dict().items()}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):expected_logits=initial_cnn(x)
    del initial_cnn;checks=[]
    for arm in ARMS:
        j=next(j for j in jobs if j['arm']==arm);m=model_for(j).eval()
        state={name:value.detach().cpu().clone() for name,value in m.state_dict().items()}
        if arm!='dino_full':
            assert all(torch.equal(value,expected[name]) for name,value in state.items())
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):assert torch.equal(m(x),expected_logits)
        flop=counted_flops(m,x[:1]);m.train()
        opt=transfer.optimizer_for(m) if arm=='dino_full' else torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.05)
        scaler=torch.amp.GradScaler('cuda',init_scale=128.)
        with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(m(x),y)
        scaler.scale(loss).backward();scaler.unscale_(opt)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        if arm=='dino_full':torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        else:assert m.layer4[-1].bn2.weight.grad.abs().sum()>0
        scaler.step(opt);scaler.update()
        if arm!='dino_full':
            bns=[(name,v) for name,v in m.named_modules() if isinstance(v,torch.nn.BatchNorm2d)];assert len(bns)==36
            for name,bn in bns:
                if arm=='resnet_bn_frozen':
                    assert not bn.training
                    for key in ['running_mean','running_var','num_batches_tracked']:assert torch.equal(m.state_dict()[name+'.'+key].cpu(),state[name+'.'+key])
                else:assert bn.num_batches_tracked.cpu()>state[name+'.num_batches_tracked']
        checks.append(dict(arm=arm,parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
                           counted_flops_224=flop,learning_rates=[group['lr'] for group in opt.param_groups]))
        del m,opt,scaler;torch.cuda.empty_cache()
    core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),new_neural_trainings=45,source_budgets=checks,
        checks=['new45/46/47 frozen folds,all45jobs paired and train-only','three-way patient isolation and2386outercoverage per source/seed',
                'pretrained/source/runtime/full-cache hashes','CNN paired initialization and eval outputs exact',
                'real batch32 finite AMP updates with original source optimizers','frozen36BNbuffers unchanged and affine gradients nonzero; ordinaryBN counters update',
                'DINO original learning rates/clipping and parameter coverage','unchanged audited R12 temperature/mixing helper; no additional tuning']))
    arch.commit('test: validate new split replication, original source updates and BN freeze behavior')
    print('PASS round13 replication preflight',flush=True)


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
        distribution=np.mean(boot,axis=0);family=np.percentile(distribution,[2.5/3,100-2.5/3]).tolist()
        comparisons[a+'_vs_'+b]=dict(seed_deltas=ds,mean_delta=float(np.mean(ds)),patient95=np.percentile(distribution,[2.5,97.5]).tolist(),family_interval=family,
                                   strong_pair=bool(min(ds)>0 and np.mean(ds)>=.01 and family[0]>0))
    strong=all(comparisons[a+'_vs_'+b]['strong_pair'] for a,b in COMPARISONS[:3])
    core.save_json(STUDY/'round13_results.json',summary);core.save_json(STUDY/'fold_results.json',folds);core.save_json(STUDY/'error_overlap.json',overlap)
    core.save_json(STUDY/'paired_comparisons.json',dict(comparisons=comparisons,strong_primary_followup=strong))
    lines=['# Round13: new-split secondary fusion replication','',
           '45 NEW source models in fixed seeds45/46/47,45 inner-only temperatures,6 unchanged pipelines. Same185 internal patients, no fixed val/test or relabeling.',
           'Mature ensemble/calibration diagnostic, not new architecture or budget matching.','',
           '| Pipeline | Mean pooled OOF F1 | Malignant recall | NLL | Brier |','|---|---:|---:|---:|---:|']
    for arm in SCHEMES:
        rr=[r for r in summary if r['scheme']==arm]
        lines.append(f'| {arm} | {np.mean([r["macro_f1"] for r in rr]):.6f} | {np.mean([r["malignant_recall"] for r in rr]):.6f} | {np.mean([r["nll"] for r in rr]):.6f} | {np.mean([r["brier"] for r in rr]):.6f} |')
    lines+=['',f'Complete preregistered primary follow-up gate: {strong}.','Intervals condition on repeated-development cross-fitted sources; not external confirmation or clinical confidence.']
    (STUDY/'ROUND13_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    arch.commit('results: replicate secondary fixed fusion with new split and training randomness')
    core.save_json(STUDY/'status.json',dict(phase='round13_complete_needs_analysis',round=13,total=45,completed=45,new_neural_trainings=45,updated_at=core.timestamp()))

arch.summarize=summarize
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','worker','run']);parser.add_argument('--job');args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='check':check()
    elif args.command=='worker':worker(args.job)
    else:arch.run()
