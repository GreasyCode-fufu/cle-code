"""Finite AC update-recipe and frozen/full DINO benchmark; nine new models."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_ac_ls_mechanisms as prior
import run_token_adapter_study as transfer
from pretrained_representations import DINO_SHA, runtime, state_sha
from models.optimize.cle_token_adapter import CLETokenAdapterNet, counted_flops
import ac_resource_scheduler
core=transfer.core
core.save_json=transfer.SAVE
STUDY=ROOT/'experiments/ac_backbone_recipe'
COHORT=prior.COHORT
core.STUDY=STUDY
core.WEIGHTS=ROOT/'weights/ac_backbone_recipe'
transfer.STUDY=STUDY
SEEDS=[45,46,47]
ARMS=['cnn_low_lr','dino_frozen','dino_full']


def sources():
    paths=dict(driver='experiments/run_ac_backbone_recipe.py',scheduler='experiments/ac_resource_scheduler.py',
        protocol='experiments/AC_BACKBONE_RECIPE_PROTOCOL.md',transfer='experiments/run_token_adapter_study.py',
        token_model='models/optimize/cle_token_adapter.py',dino_loader='experiments/pretrained_representations.py',
        core='experiments/run_data_study.py',resnet='models/Resnet.py',common='cle_fair_common.py',
        prior='experiments/run_ac_ls_mechanisms.py',cohort='experiments/run_ac_dataset.py',resource='experiments/run_ac_architecture.py',
        license='experiments/DINOV2_LICENSE.txt')
    return dict(files={k:core.digest(ROOT/v) for k,v in paths.items()},runtime=runtime())


def commit(message):
    subprocess.run([core.GIT,'-C',str(ROOT),'add',str(STUDY.relative_to(ROOT)),'experiments/run_ac_backbone_recipe.py',
                    'experiments/ac_resource_scheduler.py','experiments/AC_BACKBONE_RECIPE_PROTOCOL.md','experiments/RESEARCH_LEDGER.md','.gitignore'],check=True)
    if subprocess.run([core.GIT,'-C',str(ROOT),'diff','--cached','--quiet']).returncode:
        subprocess.run([core.GIT,'-C',str(ROOT),'commit','-m',message],check=True)


def verify():
    cfg=core.read_json(STUDY/'config.json');assert cfg['sources']==sources()
    for p,k in [(COHORT/'dataset_config.json','cohort_sha256'),(STUDY/'manifest.json','manifest_sha256'),
                (STUDY/'final_jobs.json','jobs_sha256'),(prior.STUDY/'config.json','prior_config_sha256'),
                (prior.STUDY/'independent_audit.json','prior_audit_sha256')]:assert core.digest(p)==cfg[k]
    assert core.digest(core.PRETRAINED)==cfg['resnet_pretrained_sha256']
    assert core.digest(ROOT/'weights/pretrained/dinov2_vits14_pretrain.pth')==cfg['dino_pretrained_sha256']==DINO_SHA
    for p,sha in cfg['reference_hashes'].items():assert core.digest(STUDY/p)==sha
    return cfg


def model_for(seed,variant):
    import torch
    from torch import nn
    if variant=='cnn_low_lr':
        backbone=prior.previous.cohort.ORIGINAL_MODEL(seed)
        class CNNTransfer(nn.Module):
            def __init__(self):
                super().__init__();self.variant='full_finetune'
                self.encoder=nn.Sequential(backbone.conv1,backbone.bn1,backbone.relu,backbone.maxpool,
                    backbone.layer1,backbone.layer2,backbone.layer3,backbone.layer4,backbone.avgpool)
                self.head=backbone.fc
            def forward(self,x):return self.head(self.encoder(x).flatten(1))
        model=CNNTransfer().cuda()
    else:
        core.set_seed(seed)
        model=CLETokenAdapterNet(ROOT,'frozen_head' if variant=='dino_frozen' else 'full_finetune').cuda()
    torch.set_num_threads(1)
    return model


transfer.verify=verify
transfer.model_for=model_for


def prepare():
    STUDY.mkdir(parents=True,exist_ok=True)
    if (STUDY/'config.json').exists():verify();return
    assert core.read_json(prior.STUDY/'independent_audit.json')['passed']
    data=core.read_json(COHORT/'dataset_config.json');ix=data['indices']
    (STUDY/'manifest.json').write_bytes((COHORT/'manifest.json').read_bytes())
    hashes={};index=[]
    for old in core.read_json(prior.STUDY/'final_jobs.json'):
        if old['arm']!='plain':continue
        out=prior.STUDY/'runs'/old['id'];comp=core.read_json(out/'complete.json')
        sig=hashlib.sha256((json.dumps(old,sort_keys=True)+core.digest(prior.STUDY/'config.json')+core.digest(core.__file__)).encode()).hexdigest()
        assert comp['signature']==sig and core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']
        target=STUDY/'references'/f's{old["seed"]}';target.mkdir(parents=True,exist_ok=True)
        for name in ['complete.json','run_meta.json','train_log.csv','validation_predictions.csv','validation_complete.json']:
            (target/name).write_bytes((out/name).read_bytes());hashes[str((target/name).relative_to(STUDY))]=core.digest(target/name)
        index.append(dict(source_job=old,signature=sig,checkpoint=comp['checkpoint'],checkpoint_sha256=comp['checkpoint_sha256']))
    assert len(index)==3
    core.save_json(STUDY/'reference_index.json',index);hashes['reference_index.json']=core.digest(STUDY/'reference_index.json')
    jobs=[dict(id=f'acbackbone_{arm}_s{seed}',phase='ac_transfer',arm=arm,variant=arm,seed=seed,epochs=60,
        train=ix['train'],select=ix['val'],score=ix['val'],sample_weights={}) for arm in ARMS for seed in SEEDS]
    core.save_json(STUDY/'final_jobs.json',jobs)
    core.save_json(STUDY/'config.json',dict(sources=sources(),cohort_sha256=core.digest(COHORT/'dataset_config.json'),
        manifest_sha256=core.digest(STUDY/'manifest.json'),jobs_sha256=core.digest(STUDY/'final_jobs.json'),
        prior_config_sha256=core.digest(prior.STUDY/'config.json'),prior_audit_sha256=core.digest(prior.STUDY/'independent_audit.json'),
        reference_hashes=hashes,resnet_pretrained_sha256=core.digest(core.PRETRAINED),dino_pretrained_sha256=DINO_SHA,
        arms=ARMS,seeds=SEEDS,new_trainings=9,reused_references=3,initial_parallel=6,max_parallel=8,reserve_gpu_mib=6144,new_job_gpu_mib=3072,
        minimum_delta=.005,family_percentiles=[.625,99.375],bootstrap_replicates=20000,bootstrap_rng=20261024))
    commit('research: freeze AC backbone and update-recipe benchmark after failed LS replication')


def done(job):
    p=STUDY/'runs'/job['id']/'validation_complete.json'
    if not p.exists():return False
    v=core.read_json(p);assert core.complete(job) and v['signature']==core.signature(job) and v['n']==427
    assert core.digest(p.parent/'validation_predictions.csv')==v['prediction_sha256']
    return True


def worker(job_id):
    import torch
    cfg=verify();job=next(j for j in core.read_json(STUDY/'final_jobs.json') if j['id']==job_id)
    if done(job):return
    transfer.DINO_SHA=cfg['resnet_pretrained_sha256'] if job['arm']=='cnn_low_lr' else cfg['dino_pretrained_sha256']
    if not core.complete(job):transfer.train_job(job)
    out=STUDY/'runs'/job_id;comp=core.read_json(out/'complete.json');model=model_for(job['seed'],job['variant'])
    initial={k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}
    model.load_state_dict(torch.load(ROOT/comp['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    squared=count=0;changed=0
    for name,value in model.encoder.state_dict().items():
        current=value.detach().cpu();changed+=int(not torch.equal(current,initial[name]))
        if current.is_floating_point():squared+=float((current.double()-initial[name].double()).square().sum());count+=current.numel()
    assert (changed==0)==(job['arm']=='dino_frozen')
    model.eval();before=state_sha(model)
    p,y,idx=core.infer(model,core.loader(job['score'],False,seed=job['seed']))
    assert state_sha(model)==before and idx==job['score'] and abs(core.metrics(p,y)['macro_f1']-comp['best_select_f1'])<1e-12
    core.save_csv(out/'validation_predictions.csv',core.prediction_rows(p,y,idx))
    core.save_json(out/'validation_complete.json',dict(signature=core.signature(job),arm=job['arm'],seed=job['seed'],n=len(y),
        checkpoint_sha256=comp['checkpoint_sha256'],prediction_sha256=core.digest(out/'validation_predictions.csv'),
        parameters=comp['parameters'],trainable_parameters=comp['trainable_parameters'],
        source_rms_drift=(squared/count)**.5,source_tensors_changed=changed,inference_state_sha256=before,state_unchanged=True,**core.metrics(p,y)))


def check():
    import torch
    import torch.nn.functional as F
    cfg=verify();ix=core.read_json(COHORT/'dataset_config.json')['indices']
    x,y,_,_=next(iter(core.loader(ix['train'][:32],False,seed=45)));x,y=x.cuda(),y.cuda()
    original=prior.previous.cohort.new_model(45).eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):expected=original(x)
    ref=[p.detach().cpu().clone() for p in original.parameters()];del original
    frozen_initial=None;budgets=[]
    for arm in ARMS:
        model=model_for(45,arm).eval()
        if arm=='cnn_low_lr':
            assert all(torch.equal(a.detach().cpu(),b) for a,b in zip(model.parameters(),ref))
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):assert torch.equal(model(x),expected)
        elif arm=='dino_frozen':frozen_initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:assert all(torch.equal(v.detach().cpu(),frozen_initial[k]) for k,v in model.state_dict().items())
        opt=transfer.optimizer_for(model);ids=[id(p) for group in opt.param_groups for p in group['params']]
        assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
        assert [g['lr'] for g in opt.param_groups]==([1e-4] if arm=='dino_frozen' else [1e-5,1e-4])
        total=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total==(21286211 if arm=='cnn_low_lr' else 21631107)
        assert trainable==(2307 if arm=='dino_frozen' else total)
        flops=counted_flops(model,x[:1]);initial={k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}
        model.train();torch.cuda.reset_peak_memory_stats();scaler=torch.amp.GradScaler('cuda',init_scale=128.)
        with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(x),y)
        scaler.scale(loss).backward();scaler.unscale_(opt)
        params=[p for p in model.parameters() if p.requires_grad]
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
        before=[p.detach().clone() for p in model.head.parameters()];scaler.step(opt);scaler.update()
        assert any(not torch.equal(a,b) for a,b in zip(before,model.head.parameters()))
        changed=sum(not torch.equal(v.detach().cpu(),initial[k]) for k,v in model.encoder.state_dict().items())
        assert (changed==0)==(arm=='dino_frozen')
        parameter_changes=sum(not torch.equal(p.detach().cpu(),initial[k]) for k,p in model.encoder.named_parameters())
        assert (parameter_changes==0)==(arm=='dino_frozen')
        for name,value in model.encoder.state_dict().items():
            if name.endswith('num_batches_tracked'):assert int(value)==int(initial[name])+1
        budgets.append(dict(arm=arm,parameters=total,trainable_parameters=trainable,counted_flops=flops,
            peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,source_tensors_changed_after_step=changed,
            source_parameters_changed_after_step=parameter_changes,gradient_norm_before_clip=float(norm)))
        del model,opt,params,initial,before;torch.cuda.empty_cache()
    core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),budgets=budgets,
        checks=['CNN split state/AMP exactness','paired frozen/full DINO initial source/head','exact optimizer coverage/lr/trainable counts',
                'real batch32 finite AMP/clip update','frozen source unchanged/trainable source updated','parameter/FLOP/peak memory']))
    commit('test: validate AC backbone recipe initialization, optimizer and memory budgets');print(json.dumps(budgets),flush=True)


snapshot=prior.previous.snapshot
resources=prior.previous.cohort.resources

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','worker','run']);parser.add_argument('--job');args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='check':check()
    elif args.command=='worker':worker(args.job)
    else:ac_resource_scheduler.run_queue(sys.modules[__name__])
