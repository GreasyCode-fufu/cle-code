#!/usr/bin/env python3
"""Round11: full DINO adaptation patient OOF, only15 new neural models."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_token_adapter_study as transfer
core=transfer.core;arch=transfer.arch
SOURCE_TRAINING=transfer.sources()
DATA=ROOT/'experiments/data_study'
BN=ROOT/'experiments/bn_oof_study'
REP=ROOT/'experiments/representation_study'
PRIOR=ROOT/'experiments/token_adapter_study'
STUDY=ROOT/'experiments/dino_oof_study'
REFS=['resnet_bn_train','resnet_bn_frozen','dino_frozen_ridge']
ALL=['dino_full']+REFS
transfer.STUDY=STUDY;arch.STUDY=STUDY;arch.__file__=__file__
core.STUDY=STUDY;core.WEIGHTS=ROOT/'weights/dino_oof_study'


def save_json(path,value):
    if Path(path)==STUDY/'status.json' and value.get('phase')=='round1_training':
        value=dict(value,phase='round11_dino_patient_oof',round=11,new_neural_trainings=15)
    transfer.SAVE(path,value)


core.save_json=save_json
arch.commit=lambda message:transfer.COMMIT(message.replace('development-only architecture run','patient OOF full-DINO run'))


def sources():
    return dict(driver=core.digest(__file__),protocol=core.digest(ROOT/'experiments/ROUND11_PROTOCOL.md'),
                unchanged_round10_training=SOURCE_TRAINING)


def source_signature(job,source):
    return hashlib.sha256((json.dumps(job,sort_keys=True)+core.digest(source/'config.json')+core.digest(core.__file__)).encode()).hexdigest()


def verify():
    cfg=core.read_json(STUDY/'config.json')
    assert cfg['architecture_sources']==sources()
    for path,key in [(DATA/'config.json','source_data_config_sha256'),(DATA/'oof_jobs.json','source_folds_sha256'),
                     (BN/'config.json','source_bn_config_sha256'),(REP/'config.json','source_representation_config_sha256'),
                     (PRIOR/'config.json','source_round10_config_sha256'),(STUDY/'manifest.json','manifest_sha256'),
                     (STUDY/'reference_index.json','reference_index_sha256'),(REP/'completion_audit.json','reference_dino_audit_sha256'),
                     (STUDY/'final_jobs.json','jobs_sha256'),(core.CACHE/'meta.json','cache_meta_sha256')]:
        assert core.digest(path)==cfg[key]
    for relative,sha in cfg['reference_file_hashes'].items():assert core.digest(STUDY/relative)==sha
    assert core.digest(ROOT/'weights/pretrained/dinov2_vits14_pretrain.pth')==transfer.DINO_SHA
    return cfg


transfer.verify=verify;arch.verify=verify;arch.baseline_predictions=lambda:None


def validate_predictions(path,indices,records):
    rr=sorted(arch.read_csv(path),key=lambda row:int(row['index']))
    assert [int(row['index']) for row in rr]==sorted(indices)
    p=np.array([[float(row[f'p{k}']) for k in range(3)] for row in rr])
    y=np.array([int(row['label']) for row in rr])
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=2e-6)
    assert np.array_equal(p.argmax(1),[int(row['pred']) for row in rr])
    for row in rr:
        record=records[int(row['index'])]
        assert record['split']=='train' and record['label']==int(row['label'])
        assert record['sample_id']==row['sample_id'] and record['group_id']==row['group_id']
    return rr,core.metrics(p,y)


def prepare():
    STUDY.mkdir(parents=True,exist_ok=True)
    if (STUDY/'config.json').exists():verify();return
    (STUDY/'manifest.json').write_bytes((DATA/'manifest.json').read_bytes())
    originals=core.read_json(DATA/'oof_jobs.json');frozen=core.read_json(BN/'final_jobs.json')
    jobs=[];hashes={};index=[]
    for old in originals:
        assert old['phase']=='oof' and not old.get('sample_weights')
        new=dict(old,id=f'r11_dino_full_s{old["seed"]}_f{old["fold"]}',arm='dino_full',variant='full_finetune',
                 recipe='full_unweighted',source_oof_job_id=old['id'],architecture_sources=sources())
        jobs.append(new)
        for arm,source,sourcejob in [('resnet_bn_train',DATA,old),('resnet_bn_frozen',BN,next(j for j in frozen if j['source_job_id']==old['id']))]:
            out=source/'runs'/sourcejob['id'];comp=core.read_json(out/'complete.json')
            assert comp['signature']==source_signature(sourcejob,source) and comp['epochs_completed']==60
            assert core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']
            target=STUDY/'references'/arm/old['id'];target.mkdir(parents=True,exist_ok=True)
            for name in ['complete.json','run_meta.json','train_log.csv','oof_predictions.csv']:
                shutil.copyfile(out/name,target/name);hashes[str((target/name).relative_to(STUDY))]=core.digest(target/name)
            index.append(dict(arm=arm,fold=old['fold'],seed=old['seed'],source_dir=str(source.relative_to(ROOT)),
                              source_job=sourcejob,checkpoint_sha256=comp['checkpoint_sha256'],source_signature=comp['signature']))
    for seed in [42,43,44]:
        target=STUDY/'references'/f'dino_frozen_ridge_s{seed}.csv'
        shutil.copyfile(REP/'oof'/f'dino_concat_s{seed}.csv',target);hashes[str(target.relative_to(STUDY))]=core.digest(target)
    core.save_json(STUDY/'reference_index.json',index)
    core.save_json(STUDY/'final_jobs.json',jobs)
    cfg=dict(round=11,created_at=core.timestamp(),architecture_sources=sources(),schemes=ALL,seeds=[42,43,44],
             active_indices=list(range(2386)),new_neural_trainings=15,reused_cnn_models=30,reused_frozen_dino_oof_files=3,
             source_data_config_sha256=core.digest(DATA/'config.json'),source_folds_sha256=core.digest(DATA/'oof_jobs.json'),
             source_bn_config_sha256=core.digest(BN/'config.json'),source_representation_config_sha256=core.digest(REP/'config.json'),
             source_round10_config_sha256=core.digest(PRIOR/'config.json'),manifest_sha256=core.digest(STUDY/'manifest.json'),
             jobs_sha256=core.digest(STUDY/'final_jobs.json'),cache_meta_sha256=core.digest(core.CACHE/'meta.json'),
             reference_index_sha256=core.digest(STUDY/'reference_index.json'),reference_dino_audit_sha256=core.digest(REP/'completion_audit.json'),
             reference_file_hashes=hashes,primary='dino_full',comparators=REFS,epochs=60,
             recipe='Unchanged round10 full_finetune train_job; head1e-4 encoder1e-5 AdamWwd.05 cosine0 batch32 AMP clip1',
             selection='Only original inner patient indices; outer inferred after checkpoint frozen',
             bootstrap=dict(groups=185,draws=20000,seed=20261024,family_comparisons=3,percentiles=[2.5/3,100-2.5/3]),
             gate='Against EACH reference: three positive seed deltas, mean>=.01 and family lower>0',
             policy='Only original train; no grades/weights/fixed val/test/relabeling. Internal system-level cross-fitting, not external clinical validation.')
    core.save_json(STUDY/'config.json',cfg)
    core.save_json(STUDY/'status.json',dict(phase='prepared',round=11,total=15,updated_at=core.timestamp()))
    arch.commit('research: freeze full DINO patient cross-fitting and signed CNN/frozen-feature references')


def worker(job_id):
    import torch
    verify();job=next(j for j in core.read_json(STUDY/'final_jobs.json') if j['id']==job_id)
    if not core.complete(job):transfer.train_job(job)
    out=STUDY/'runs'/job_id;comp=core.read_json(out/'complete.json')
    model=transfer.model_for(job['seed'],'full_finetune')
    model.load_state_dict(torch.load(ROOT/comp['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    pi,yi,ii=core.infer(model,core.loader(job['select'],False,seed=job['seed']))
    assert ii==job['select'] and abs(core.metrics(pi,yi)['macro_f1']-comp['best_select_f1'])<1e-12
    core.save_csv(out/'inner_predictions.csv',core.prediction_rows(pi,yi,ii))
    p,y,idx=core.infer(model,core.loader(job['score'],False,seed=job['seed']))
    assert idx==job['score']
    core.save_csv(out/'oof_predictions.csv',core.prediction_rows(p,y,idx))
    rr,metrics=validate_predictions(out/'oof_predictions.csv',job['score'],core.read_json(STUDY/'manifest.json'))
    core.save_json(out/'outer_complete.json',dict(signature=core.signature(job),job_id=job_id,n=len(rr),
        checkpoint_sha256=comp['checkpoint_sha256'],prediction_sha256=core.digest(out/'oof_predictions.csv'),
        inner_prediction_sha256=core.digest(out/'inner_predictions.csv'),completed_at=core.timestamp(),**metrics))


def done(job):
    path=STUDY/'runs'/job['id']/'outer_complete.json'
    if not path.exists():return False
    assert core.complete(job)
    result=core.read_json(path)
    assert result['signature']==core.signature(job) and result['n']==len(job['score'])
    assert core.digest(path.parent/'oof_predictions.csv')==result['prediction_sha256']
    assert core.digest(path.parent/'inner_predictions.csv')==result['inner_prediction_sha256']
    return True


arch.done=done


def check():
    import torch
    import torch.nn.functional as F
    cfg=verify();records=core.read_json(STUDY/'manifest.json');jobs=core.read_json(STUDY/'final_jobs.json')
    original=core.read_json(DATA/'oof_jobs.json');coverage=collections.Counter()
    assert len(jobs)==15
    assert core.digest(core.CACHE/'resized_uint8.npy')==core.read_json(core.CACHE/'meta.json')['cache_sha256']
    for job in jobs:
        old=next(j for j in original if j['id']==job['source_oof_job_id'])
        for key in ['train','select','score','seed','fold','epochs']:assert old[key]==job[key]
        assert not job.get('sample_weights')
        assert all(0<=i<2386 for key in ['train','select','score'] for i in job[key])
        core.validate_split(records,job['train'],job['select'],job['score'])
        coverage.update((job['seed'],i) for i in job['score'])
    assert set(coverage)=={(seed,i) for seed in [42,43,44] for i in range(2386)} and set(coverage.values())=={1}
    for entry in core.read_json(STUDY/'reference_index.json'):
        source=ROOT/entry['source_dir'];old=entry['source_job']
        target=STUDY/'references'/entry['arm']/f'oof_s{old["seed"]}_f{old["fold"]}'
        comp=core.read_json(target/'complete.json')
        assert comp['signature']==entry['source_signature']==source_signature(old,source)
        assert core.digest(ROOT/comp['checkpoint'])==entry['checkpoint_sha256']
        assert [int(row['epoch']) for row in arch.read_csv(target/'train_log.csv')]==list(range(60))
        rr,metric=validate_predictions(target/'oof_predictions.csv',old['score'],records)
        assert abs(metric['macro_f1']-comp['outer_metrics']['macro_f1'])<1e-12
    for seed in cfg['seeds']:
        target=STUDY/'references'/f'dino_frozen_ridge_s{seed}.csv'
        assert core.digest(target)==core.digest(REP/'oof'/f'dino_concat_s{seed}.csv')
        rr,metric=validate_predictions(target,range(2386),records)
        old=next(row for row in core.read_json(REP/'round9_results.json') if row['scheme']=='dino_concat' and row['seed']==seed)
        assert abs(metric['macro_f1']-old['macro_f1'])<1e-12
    job=jobs[0];x,y,_,_=next(iter(core.loader(job['train'][:32],False,seed=42)));x,y=x.cuda(),y.cuda()
    m=transfer.model_for(42,'full_finetune');expected={key:value.detach().cpu().clone() for key,value in m.state_dict().items()}
    m.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):first=m(x)
    second=transfer.model_for(42,'full_finetune').eval()
    assert all(torch.equal(value.detach().cpu(),expected[key]) for key,value in second.state_dict().items())
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):assert torch.equal(first,second(x))
    del second
    opt=transfer.optimizer_for(m)
    assert [group['lr'] for group in opt.param_groups]==[1e-5,1e-4]
    assert set(id(p) for group in opt.param_groups for p in group['params'])==set(id(p) for p in m.parameters())
    m.train();scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(m(x),y)
    scaler.scale(loss).backward();scaler.unscale_(opt)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.encoder.patch_embed.proj.weight.grad.abs().sum()>0 and m.head.weight.grad.abs().sum()>0
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
    scaler.step(opt);scaler.update()
    assert not torch.equal(m.head.weight.detach().cpu(),expected['head.weight'])
    core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),new_neural_trainings=15,
        reused_cnn_models=30,reused_dino_ridge_oof=3,trainable_parameters=sum(p.numel() for p in m.parameters()),
        checks=['exact15 original inner/outer patient splits, only original train indices','2386 outer predictions per seed, patient-disjoint three sets',
                '30 reused checkpoint signatures/SHA/60ep/outer metrics and 3 frozen-DINO OOF hashes',
                'full input cache SHA with loader-restricted scope','unchanged round10 source/optimizer kernel and paired initialization',
                'encoder/head learning rates and exact optimizer coverage','real batch32 finite AMP update with gradient clipping and source/head gradients'],
        gradient_norm_before_clip=float(norm)))
    arch.commit('test: validate full-DINO OOF scope, reference provenance and unchanged adaptation kernel')
    print('PASS round11 patient OOF preflight',flush=True)


def summarize():
    cfg=verify();records=core.read_json(STUDY/'manifest.json');jobs=core.read_json(STUDY/'final_jobs.json')
    predictions={};folds=[];summary=[]
    for arm in ALL:
        for seed in cfg['seeds']:
            merged=[]
            if arm=='dino_frozen_ridge':
                merged,_=validate_predictions(STUDY/'references'/f'dino_frozen_ridge_s{seed}.csv',range(2386),records)
            else:
                for job in [j for j in jobs if j['seed']==seed]:
                    if arm=='dino_full':assert done(job);out=STUDY/'runs'/job['id']
                    else:out=STUDY/'references'/arm/job['source_oof_job_id']
                    rr,metric=validate_predictions(out/'oof_predictions.csv',job['score'],records);merged+=rr
                    folds.append(dict(scheme=arm,seed=seed,fold=job['fold'],n=len(rr),**metric))
            merged=sorted(merged,key=lambda row:int(row['index']));assert [int(row['index']) for row in merged]==list(range(2386))
            core.save_csv(STUDY/'oof'/f'{arm}_s{seed}.csv',merged);predictions[arm,seed]=merged
            p=np.array([[float(row[f'p{k}']) for k in range(3)] for row in merged]);y=np.array([int(row['label']) for row in merged])
            metric=core.metrics(p,y);groups=sorted({row['group_id'] for row in merged});cm=np.array(metric['confusion_matrix'])
            patient_acc=np.mean([np.mean([row['pred']==row['label'] for row in merged if row['group_id']==g]) for g in groups])
            summary.append(dict(scheme=arm,seed=seed,n=2386,malignant_recall=float(cm[1,1]/cm[1].sum()),patient_mean_accuracy=float(patient_acc),**metric))
    weights=np.random.default_rng(20261024).multinomial(len(groups),np.ones(len(groups))/len(groups),size=20000)
    def matrix(rr):
        m=np.zeros((len(groups),3,3));np.add.at(m,([groups.index(row['group_id']) for row in rr],[int(row['label']) for row in rr],[int(row['pred']) for row in rr]),1);return m
    def f1(m):
        den=m.sum(-1)+m.sum(-2);return np.mean(np.divide(2*np.diagonal(m,axis1=-2,axis2=-1),den,out=np.zeros_like(den),where=den!=0),axis=-1)
    comparisons={}
    for arm in REFS:
        ds=[];boot=[]
        for seed in cfg['seeds']:
            a,b=matrix(predictions[arm,seed]),matrix(predictions['dino_full',seed])
            ds.append(float(f1(b.sum(0))-f1(a.sum(0))))
            boot.append(f1(np.einsum('bg,gij->bij',weights,b))-f1(np.einsum('bg,gij->bij',weights,a)))
        distribution=np.mean(boot,axis=0);family=np.percentile(distribution,[2.5/3,100-2.5/3]).tolist()
        comparisons['dino_full_vs_'+arm]=dict(seed_deltas=ds,mean_delta=float(np.mean(ds)),patient95=np.percentile(distribution,[2.5,97.5]).tolist(),
            family_interval=family,strong_pair=bool(min(ds)>0 and np.mean(ds)>=.01 and family[0]>0))
    strong=all(row['strong_pair'] for row in comparisons.values())
    core.save_json(STUDY/'round11_results.json',summary);core.save_json(STUDY/'fold_results.json',folds)
    core.save_json(STUDY/'paired_comparisons.json',dict(comparisons=comparisons,strong_primary_followup=strong))
    lines=['# Round11: patient OOF check of full DINO adaptation','',
           '15 new full-DINO neural models,30 reused CNNs,3 frozen-DINO OOF reference files. Original train only; fixed val/test excluded.',
           'Cross-fitted internal system comparisons; not a new architecture or external confirmation.','',
           '| Pipeline | Seed | Pooled OOF F1 | Malignant recall | Patient mean accuracy |','|---|---:|---:|---:|---:|']
    for row in summary:lines.append(f'| {row["scheme"]} | {row["seed"]} | {row["macro_f1"]:.6f} | {row["malignant_recall"]:.6f} | {row["patient_mean_accuracy"]:.6f} |')
    lines+=['',f'Complete preregistered internal follow-up gate: {strong}.','Conditional intervals do not cover all overlapping-training/selection/pretraining confounds.']
    (STUDY/'ROUND11_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    arch.commit('results: compare full DINO adaptation on paired patient OOF against three frozen references')
    core.save_json(STUDY/'status.json',dict(phase='round11_complete_needs_analysis',round=11,completed=15,total=15,updated_at=core.timestamp()))


arch.summarize=summarize
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','worker','run']);parser.add_argument('--job');args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='check':check()
    elif args.command=='worker':worker(args.job)
    else:arch.run()
