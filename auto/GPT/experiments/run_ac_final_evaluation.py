"""One frozen common AC test evaluation, zero neural training."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_ac_backbone_recipe as third
import run_ac_architecture as first
import run_ac_ls_mechanisms as second
from pretrained_representations import state_sha,runtime
from predict_ac_model import load_model,preprocess
core=third.core;STUDY=ROOT/'experiments/ac_final_evaluation';COHORT=third.COHORT
core.STUDY=STUDY


def sources():
    return dict(files={p:core.digest(ROOT/p) for p in ['experiments/run_ac_final_evaluation.py','experiments/predict_ac_model.py',
        'experiments/AC_FINAL_EVALUATION_PROTOCOL.md','experiments/run_ac_dataset.py','experiments/run_ac_architecture.py',
        'experiments/run_ac_ls_mechanisms.py','experiments/run_ac_backbone_recipe.py','experiments/run_token_adapter_study.py',
        'experiments/pretrained_representations.py','experiments/run_data_study.py','models/optimize/cle_morphology.py',
        'models/optimize/cle_ls_mechanisms.py','models/optimize/cle_token_adapter.py','models/Resnet.py','cle_fair_common.py']},runtime=runtime())


def signature(job):return hashlib.sha256((json.dumps(job,sort_keys=True)+core.digest(STUDY/'config.json')).encode()).hexdigest()


def commit(message):
    subprocess.run([core.GIT,'-C',str(ROOT),'add',str(STUDY.relative_to(ROOT)),'experiments/run_ac_final_evaluation.py',
        'experiments/predict_ac_model.py','experiments/AC_FINAL_EVALUATION_PROTOCOL.md','experiments/RESEARCH_LEDGER.md','.gitignore'],check=True)
    if subprocess.run([core.GIT,'-C',str(ROOT),'diff','--cached','--quiet']).returncode:
        subprocess.run([core.GIT,'-C',str(ROOT),'commit','-m',message],check=True)


def verify():
    cfg=core.read_json(STUDY/'config.json');assert cfg['sources']==sources()
    for p,sha in cfg['frozen_hashes'].items():assert core.digest(ROOT/p)==sha
    assert core.digest(STUDY/'manifest.json')==cfg['manifest_sha256']
    assert core.digest(STUDY/'final_jobs.json')==cfg['jobs_sha256']
    assert core.digest(STUDY/'deployment_manifest.json')==cfg['deployment_sha256']
    return cfg


def model_for(job):
    old=job['source_job'];kind=job['source_kind']
    if kind=='baseline':return first.cohort.new_model(old['seed'])
    if kind=='round1':return first.model_for(old['seed'],old['variant'],old['control_width'])
    if kind=='round2':return second.model_for(old['seed'],old['arm'],old['control_width'])
    return third.model_for(old['seed'],old['arm'])


def prepare():
    STUDY.mkdir(parents=True,exist_ok=True)
    if (STUDY/'config.json').exists():verify();return
    first.verify();second.verify();third.verify()
    for folder in [first.STUDY,second.STUDY,third.STUDY]:assert core.read_json(folder/'independent_audit.json')['passed']
    data=core.read_json(COHORT/'dataset_config.json');score=data['indices']['test'];assert len(score)==576
    (STUDY/'manifest.json').write_bytes((COHORT/'manifest.json').read_bytes())
    jobs=[];frozen={}
    for kind,folder in [('baseline',first.BASELINE),('round1',first.STUDY),('round2',second.STUDY),('round3',third.STUDY)]:
        frozen[str((folder/'config.json').relative_to(ROOT))]=core.digest(folder/'config.json')
        for old in core.read_json(folder/'final_jobs.json'):
            out=folder/'runs'/old['id'];comp=core.read_json(out/'complete.json');validation=core.read_json(out/'validation_complete.json')
            sig=hashlib.sha256((json.dumps(old,sort_keys=True)+core.digest(folder/'config.json')+core.digest(core.__file__)).encode()).hexdigest()
            assert comp['signature']==sig and core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']
            assert old['train']==data['indices']['train'] and old['score']==data['indices']['val'] and comp['epochs_completed']==60
            arm=old['arm']
            family=('cnn_original' if kind=='baseline' or (kind=='round2' and arm=='plain') else
                    'ls_signed' if arm=='ls_reference' else 'conv_small' if arm=='conv_control' else arm)
            jobs.append(dict(id=f'final_{kind}_{old["id"]}',source_kind=kind,source_folder=str(folder.relative_to(ROOT)),
                source_job=old,source_signature=sig,checkpoint=comp['checkpoint'],checkpoint_sha256=comp['checkpoint_sha256'],
                validation_f1=validation['macro_f1'],family=family,seed=old['seed'],score=score))
            frozen[str((out/'complete.json').relative_to(ROOT))]=core.digest(out/'complete.json')
            frozen[str((out/'validation_predictions.csv').relative_to(ROOT))]=core.digest(out/'validation_predictions.csv')
    assert len(jobs)==39 and len({j['checkpoint_sha256'] for j in jobs})==39
    for folder in [first.STUDY,second.STUDY,third.STUDY]:frozen[str((folder/'independent_audit.json').relative_to(ROOT))]=core.digest(folder/'independent_audit.json')
    frozen[str((COHORT/'dataset_config.json').relative_to(ROOT))]=core.digest(COHORT/'dataset_config.json')
    deploy=ROOT/'weights/ac_final_evaluation/deployment';deploy.mkdir(parents=True,exist_ok=True);models={}
    for name,family in [('dino','dino_full'),('cnn','cnn_original')]:
        job=next(j for j in jobs if j['family']==family and j['seed']==45)
        destination=deploy/(name+'_s45.pth');shutil.copy2(ROOT/job['checkpoint'],destination)
        assert core.digest(destination)==job['checkpoint_sha256']
        models[name]=dict(final_job_id=job['id'],checkpoint=str(destination.relative_to(ROOT)),checkpoint_sha256=job['checkpoint_sha256'],seed=45,
            validation_f1=job['validation_f1'],selection='First preregistered new seed; not best seed and never selected on test.')
    records=core.read_json(STUDY/'manifest.json')
    class_names=[next(r['class_name'] for r in records if r['label']==k) for k in range(3)]
    core.save_json(STUDY/'deployment_manifest.json',dict(class_names=class_names,models=models,default='dino',
        policy='Higher validation-mean exploratory DINO candidate; validation family gate failed. CNN compute alternative. No test reselection.'))
    core.save_json(STUDY/'final_jobs.json',jobs)
    core.save_json(STUDY/'config.json',dict(sources=sources(),frozen_hashes=frozen,manifest_sha256=core.digest(STUDY/'manifest.json'),
        jobs_sha256=core.digest(STUDY/'final_jobs.json'),deployment_sha256=core.digest(STUDY/'deployment_manifest.json'),
        new_neural_trainings=0,models=39,test_images=576,test_patients=41,primary='dino_full',primary_seeds=[45,46,47],
        references=['cnn_original','cnn_low_lr','dino_frozen'],bootstrap_draws=20000,bootstrap_rng=20261024,
        family_percentiles=[2.5/3,100-2.5/3],minimum_delta=.005,default_model='dino',test_reselection_allowed=False))
    commit('research: freeze all 39 AC checkpoints and exploratory final candidates before common test')


def check():
    import torch
    cfg=verify();jobs=core.read_json(STUDY/'final_jobs.json');manifest=core.read_json(STUDY/'manifest.json');ix=jobs[0]['source_job']['score'][:32]
    loader=core.loader(ix,False,seed=45);x,_,_,ids=next(iter(loader))
    external=torch.stack([preprocess(ROOT/manifest[i]['path']) for i in ids.tolist()]);assert torch.equal(x,external)
    results=[]
    for name,item in core.read_json(STUDY/'deployment_manifest.json')['models'].items():
        job=next(j for j in jobs if j['id']==item['final_job_id']);original=model_for(job)
        original.load_state_dict(torch_load(ROOT/job['checkpoint']),strict=True);original.eval()
        api,_=load_model(STUDY/'deployment_manifest.json',name)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            p=original(x.cuda());q=api(x.cuda());assert torch.equal(p,q)
        assert all(torch.isfinite(p).all() for p in api.parameters())
        results.append(dict(kind=name,validation_batch=32,preprocess_exact=True,amp_logits_exact=True))
        del api,original;torch.cuda.empty_cache()
    core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),test_images_loaded=0,checks=results))
    commit('test: validate standalone frozen AC inference interface on validation images only')


def torch_load(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=True)


def done(job):
    p=STUDY/'runs'/job['id']/'test_complete.json'
    if not p.exists():return False
    v=core.read_json(p);assert v['signature']==signature(job) and v['checkpoint_sha256']==job['checkpoint_sha256'] and v['n']==576
    assert core.digest(p.parent/'test_predictions.csv')==v['prediction_sha256']
    return True


def worker(job_id):
    import torch
    verify();job=next(j for j in core.read_json(STUDY/'final_jobs.json') if j['id']==job_id)
    if done(job):return
    assert core.digest(ROOT/job['checkpoint'])==job['checkpoint_sha256']
    model=model_for(job);model.load_state_dict(torch_load(ROOT/job['checkpoint']),strict=True);model.eval().requires_grad_(False)
    before=state_sha(model);p,y,idx=core.infer(model,core.loader(job['score'],False,seed=job['seed']))
    assert idx==job['score'] and state_sha(model)==before
    out=STUDY/'runs'/job_id;out.mkdir(parents=True,exist_ok=True)
    core.save_csv(out/'test_predictions.csv',core.prediction_rows(p,y,idx))
    core.save_json(out/'test_complete.json',dict(signature=signature(job),family=job['family'],seed=job['seed'],n=len(y),
        checkpoint_sha256=job['checkpoint_sha256'],prediction_sha256=core.digest(out/'test_predictions.csv'),state_sha256=before,
        state_unchanged=True,**core.metrics(p,y)))


def run():
    import csv
    with (STUDY/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify();assert core.read_json(STUDY/'preflight_validation.json')['passed']
        jobs=core.read_json(STUDY/'final_jobs.json');pending=[j for j in jobs if not done(j)];running={};logs=STUDY/'logs';logs.mkdir(exist_ok=True)
        while pending or running:
            finished=[]
            for pid,(p,j,f) in list(running.items()):
                if p.poll() is None:continue
                f.close();del running[pid]
                assert p.returncode==0 and done(j),(j['id'],'inspect logs; reuse valid inference')
                finished.append(j['id'])
            while pending and len(running)<6 and third.resources()['gpu_free_mib']>8192:
                j=pending.pop(0);f=(logs/(j['id']+'.log')).open('a',buffering=1)
                p=subprocess.Popen([sys.executable,__file__,'worker','--job',j['id']],cwd=ROOT,
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                running[p.pid]=(p,j,f)
            stats=third.snapshot();stats.update(running=len(running),pending=len(pending));path=STUDY/'resource_samples.csv';exists=path.exists()
            with path.open('a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(stats))
                if not exists:w.writeheader()
                w.writerow(stats)
            if finished:commit('results: complete frozen AC test inference '+','.join(finished))
            core.save_json(STUDY/'status.json',dict(phase='frozen_test_inference',total=39,completed=39-len(pending)-len(running),
                running=[dict(job=j['id'],pid=p.pid) for p,j,_ in running.values()],resources=stats,updated_at=core.timestamp()))
            if pending or running:time.sleep(2)
        core.save_json(STUDY/'results.json',[core.read_json(STUDY/'runs'/j['id']/'test_complete.json') for j in jobs])
        commit('results: complete single frozen 39-model AC test evaluation for independent audit')
        core.save_json(STUDY/'status.json',dict(phase='complete_needs_independent_audit',completed=39,total=39,updated_at=core.timestamp()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','worker','run']);parser.add_argument('--job');args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='check':check()
    elif args.command=='worker':worker(args.job)
    else:run()
