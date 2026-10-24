#!/usr/bin/env python3
"""Controlled architecture design on the user-requested frozen AC cohort."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_ac_dataset as cohort
from models.optimize.cle_morphology import MorphologyResNet,CLEContext,PlainControl
core=cohort.core;arch=cohort.arch
COHORT=cohort.STUDY;BASELINE=cohort.MODEL_STUDY;STUDY=ROOT/'experiments/ac_architecture_v1'
MODEL=ROOT/'models/optimize/cle_morphology.py';VARIANTS=['conv_control','ls_reference','cle_context','cle_no_contrast'];SEEDS=[42,43,44]
core.STUDY=STUDY;core.WEIGHTS=ROOT/'weights/ac_architecture_v1'
CPU_PREVIOUS=None


def sources():
    return dict(driver=core.digest(__file__),model=core.digest(MODEL),trainer=core.digest(core.__file__),
        cohort_driver=core.digest(ROOT/'experiments/run_ac_dataset.py'),protocol=core.digest(ROOT/'experiments/AC_ARCHITECTURE_V1_PROTOCOL.md'))


def commit(message):
    subprocess.run([core.GIT,'-C',str(ROOT),'add','experiments/ac_architecture_v1','experiments/run_ac_architecture.py',
                    'experiments/AC_ARCHITECTURE_V1_PROTOCOL.md','experiments/audit_ac_dataset.py',
                    'experiments/ac_dataset_v1/independent_dataset_audit.json','experiments/RESEARCH_LEDGER.md','.gitignore'],check=True)
    if subprocess.run([core.GIT,'-C',str(ROOT),'diff','--cached','--quiet']).returncode:subprocess.run([core.GIT,'-C',str(ROOT),'commit','-m',message],check=True)


def verify():
    cfg=core.read_json(STUDY/'config.json');assert cfg['sources']==sources()
    for p,k in [(COHORT/'dataset_config.json','cohort_sha256'),(COHORT/'independent_dataset_audit.json','cohort_audit_sha256'),
                (STUDY/'manifest.json','manifest_sha256'),(STUDY/'final_jobs.json','jobs_sha256'),(BASELINE/'config.json','baseline_config_sha256')]:assert core.digest(p)==cfg[k]
    for name,sha in cfg['baseline_hashes'].items():assert core.digest(STUDY/name)==sha
    return cfg


def model_for(seed,variant,width):
    import torch
    model=MorphologyResNet(cohort.ORIGINAL_MODEL(seed),variant,3,width).cuda();torch.set_num_threads(1)
    return model


def prepare():
    STUDY.mkdir(parents=True,exist_ok=True)
    if (STUDY/'config.json').exists():verify();return
    assert core.read_json(COHORT/'independent_dataset_audit.json')['passed']
    data=core.read_json(COHORT/'dataset_config.json');indices=data['indices'];assert [len(indices[k]) for k in ['train','val','test']]==[2077,427,576]
    (STUDY/'manifest.json').write_bytes((COHORT/'manifest.json').read_bytes())
    target=sum(p.numel() for p in CLEContext(256).parameters());candidates={w:sum(p.numel() for p in PlainControl(256,w).parameters()) for w in range(32,257,8)}
    width=min(candidates,key=lambda w:abs(candidates[w]-target));hashes={}
    for old in core.read_json(BASELINE/'final_jobs.json'):
        source=BASELINE/'runs'/old['id'];out=STUDY/'baselines'/f's{old["seed"]}';out.mkdir(parents=True,exist_ok=True)
        comp=core.read_json(source/'complete.json');signature=hashlib.sha256((json.dumps(old,sort_keys=True)+core.digest(BASELINE/'config.json')+core.digest(core.__file__)).encode()).hexdigest()
        assert comp['signature']==signature and core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']
        for name in ['complete.json','run_meta.json','train_log.csv','validation_predictions.csv','validation_complete.json']:
            (out/name).write_bytes((source/name).read_bytes());hashes[str((out/name).relative_to(STUDY))]=core.digest(out/name)
    jobs=[]
    for variant in VARIANTS:
        for seed in SEEDS:jobs.append(dict(id=f'acarch1_{variant}_s{seed}',phase='architecture',arm=variant,variant=variant,
            seed=seed,epochs=60,train=indices['train'],select=indices['val'],score=indices['val'],sample_weights={},position=3,control_width=width))
    core.save_json(STUDY/'final_jobs.json',jobs)
    cfg=dict(sources=sources(),cohort_sha256=core.digest(COHORT/'dataset_config.json'),cohort_audit_sha256=core.digest(COHORT/'independent_dataset_audit.json'),
        manifest_sha256=core.digest(STUDY/'manifest.json'),jobs_sha256=core.digest(STUDY/'final_jobs.json'),baseline_config_sha256=core.digest(BASELINE/'config.json'),
        baseline_hashes=hashes,pretrained_sha256=core.digest(core.PRETRAINED),seeds=SEEDS,variants=VARIANTS,control_width=width,
        candidate_modifier_parameters=target,control_modifier_parameters=candidates[width],parameter_relative_error=abs(candidates[width]-target)/target,
        new_trainings=12,reused_baselines=3,initial_parallel=6,max_parallel=8,reserve_gpu_mib=6144,new_job_gpu_mib=2048,
        test_policy='Only frozenAC576 after final candidate freeze; model choice uses ACval427.')
    core.save_json(STUDY/'config.json',cfg);commit('research: freeze first controlled structure round on common AC-only cohort')


def done(job):
    p=STUDY/'runs'/job['id']/'validation_complete.json'
    if not p.exists():return False
    value=core.read_json(p);assert core.complete(job) and value['signature']==core.signature(job) and value['n']==427
    for name,sha in value['file_hashes'].items():assert core.digest(p.parent/name)==sha
    return True


def worker(job_id):
    import torch
    verify();job=next(j for j in core.read_json(STUDY/'final_jobs.json') if j['id']==job_id)
    if done(job):return
    core.new_model=lambda seed:model_for(seed,job['variant'],job['control_width'])
    if not core.complete(job):core.run_worker(job_id)
    out=STUDY/'runs'/job_id;comp=core.read_json(out/'complete.json');model=core.new_model(job['seed'])
    model.load_state_dict(torch.load(ROOT/comp['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    perturb=[]
    def hook(module,inputs,output):
        perturb.extend(((output-inputs[0]).abs().flatten(1).mean(1)/(inputs[0].abs().flatten(1).mean(1)+1e-8)).detach().cpu().float().tolist())
    handle=model.modifier.register_forward_hook(hook);p,y,idx=core.infer(model,core.loader(job['score'],False,seed=job['seed']));handle.remove()
    assert abs(core.metrics(p,y)['macro_f1']-comp['best_select_f1'])<1e-12
    core.save_csv(out/'validation_predictions.csv',core.prediction_rows(p,y,idx));model.modifier.enabled=False
    off,yo,io=core.infer(model,core.loader(job['score'],False,seed=job['seed']));assert idx==io and np.array_equal(y,yo)
    core.save_csv(out/'off_predictions.csv',core.prediction_rows(off,yo,io))
    diag=dict(stream_change_mean=float(np.mean(perturb)),off_prediction_disagreement=float(np.mean(p.argmax(1)!=off.argmax(1))),
              off_metrics=core.metrics(off,yo),note='Same-checkpoint diagnostic; retrained no-contrast is a separate arm.')
    core.save_json(out/'validation_complete.json',dict(signature=core.signature(job),variant=job['variant'],seed=job['seed'],n=len(y),
        parameters=sum(v.numel() for v in model.parameters()),modifier_parameters=sum(v.numel() for v in model.modifier.parameters()),diagnostics=diag,
        file_hashes={n:core.digest(out/n) for n in ['validation_predictions.csv','off_predictions.csv']},**core.metrics(p,y)))


def snapshot():
    global CPU_PREVIOUS
    rows={line.split()[0]:list(map(int,line.split()[1:9])) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('cpu')}
    percentages={};iowait=0.
    if CPU_PREVIOUS:
        for key,v in rows.items():
            d=[b-a for a,b in zip(CPU_PREVIOUS[key],v)];total=sum(d)
            percentages[key]=100*(total-d[3]-d[4])/total if total else 0.
            if key=='cpu':iowait=100*d[4]/total if total else 0.
    CPU_PREVIOUS=rows;gpu=cohort.resources();mem={r.split(':')[0]:int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith(('MemTotal:','MemAvailable:'))}
    return dict(gpu,cpu_percent=percentages.get('cpu',0.),cpu_max_core_percent=max((v for k,v in percentages.items() if k!='cpu'),default=0.),
                cpu_iowait_percent=iowait,ram_used_percent=100*(1-mem['MemAvailable']/mem['MemTotal']))


def check():
    import torch
    import torch.nn.functional as F
    from models.optimize.cle_token_adapter import counted_flops
    cfg=verify();indices=core.read_json(COHORT/'dataset_config.json')['indices']['train'];x,y,_,_=next(iter(core.loader(indices[:32],False,seed=42)));x,y=x.cuda(),y.cuda()
    expected=cohort.new_model(42).eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):reference=expected(x)
    initial={k:v.detach().cpu().clone() for k,v in expected.state_dict().items()};del expected;budgets=[]
    for variant in VARIANTS:
        m=model_for(42,variant,cfg['control_width']).eval();assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in m.backbone.state_dict().items())
        m.modifier.enabled=False
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):assert torch.equal(m(x),reference)
        m.modifier.enabled=True;flops=counted_flops(m,x[:1]);torch.cuda.reset_peak_memory_stats();m.train();optimizer=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.05);scaler=torch.amp.GradScaler('cuda',init_scale=128.)
        with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(m(x),y)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.modifier.parameters())
        optimizer_before={k:v.detach().clone() for k,v in m.modifier.state_dict().items()};scaler.step(optimizer);scaler.update()
        assert any(not torch.equal(v,m.modifier.state_dict()[k]) for k,v in optimizer_before.items())
        budgets.append(dict(variant=variant,parameters=sum(p.numel() for p in m.parameters()),modifier_parameters=sum(p.numel() for p in m.modifier.parameters()),
                            counted_flops=flops,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20))
        del m,optimizer;torch.cuda.empty_cache()
    core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),budgets=budgets,
        checks=['same frozen AC cohort,427val/576test lists','paired source backbone and head initial states','disabled modifier equals ordinary network AMP logits',
                'realbatch32 finite AMP branch gradients and updates','parameter/FLOP/peak-memory budgets; no strict equal compute claim']))
    commit('test: validate AC morphology candidates and real GPU memory budgets');print(json.dumps(budgets),flush=True)


def run():
    import csv
    with open(STUDY/'pipeline.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg=verify();assert core.read_json(STUDY/'preflight_validation.json')['passed']
        jobs=core.read_json(STUDY/'final_jobs.json');pending=[j for j in jobs if not done(j)];running={};retries=collections.Counter();logs=STUDY/'logs';logs.mkdir(exist_ok=True)
        slots=cfg['initial_parallel'];started=time.monotonic();next_sample=0;recent=[];stats=None
        while pending or running:
            completed=[]
            for pid,(p,j,f) in list(running.items()):
                if p.poll() is None:continue
                f.close();del running[pid]
                if p.returncode or not done(j):
                    retries[j['id']]+=1
                    if retries[j['id']]<=1:pending.append(j)
                    else:raise RuntimeError(j['id']+' failed twice; inspect log and remaining children')
                else:completed.append(j['id']);print('[done]',j['id'],flush=True)
            if time.monotonic()>=next_sample:
                stats=snapshot();stats.update(running=len(running),pending=len(pending),slots=slots);recent.append(stats);recent=recent[-15:]
                path=STUDY/'resource_samples.csv';exists=path.exists()
                with path.open('a',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(stats))
                    if not exists:writer.writeheader()
                    writer.writerow(stats)
                if time.monotonic()-started>30 and len(recent)>=10 and pending and len(running)>=slots:
                    if np.mean([v['gpu_util'] for v in recent])<80 and np.mean([v['cpu_percent'] for v in recent])<75 and stats['gpu_free_mib']>cfg['reserve_gpu_mib']+cfg['new_job_gpu_mib']:
                        slots=min(slots+1,cfg['max_parallel'])
                next_sample=time.monotonic()+2
            while pending and len(running)<slots:
                free=cohort.resources()['gpu_free_mib']
                if free<cfg['reserve_gpu_mib']+cfg['new_job_gpu_mib'] or (running and stats['cpu_percent']>90):break
                job=pending.pop(0);f=open(logs/(job['id']+'.log'),'a',buffering=1);env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
                p=subprocess.Popen([sys.executable,__file__,'worker','--job',job['id']],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                running[p.pid]=(p,job,f);print('[launch]',job['id'],p.pid,flush=True)
            if completed:commit('results: complete AC architecture jobs '+','.join(completed))
            core.save_json(STUDY/'status.json',dict(phase='training',total=12,completed=12-len(pending)-len(running),pending=len(pending),slots=slots,
                running=[dict(job=j['id'],pid=p.pid) for p,j,_ in running.values()],resources=stats,updated_at=core.timestamp()))
            if pending or running:time.sleep(.5)
        results=[core.read_json(STUDY/'runs'/j['id']/'validation_complete.json') for j in jobs]
        core.save_json(STUDY/'results.json',results);commit('results: complete first AC architecture round for analysis and independent audit')
        core.save_json(STUDY/'status.json',dict(phase='complete_needs_independent_audit',new_models=12,reused_baselines=3,updated_at=core.timestamp()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','worker','run']);parser.add_argument('--job');args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='check':check()
    elif args.command=='worker':worker(args.job)
    else:run()
