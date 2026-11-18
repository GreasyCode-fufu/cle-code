#!/usr/bin/env python3
"""Separate, development-only CLE architecture study. Never evaluates test.

Reuses the immutable data-study training implementation and cache. Every run
adds hashes for this driver and the model source to its artifact signature.
"""
import argparse
import collections
import csv
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments'))
import run_data_study as core
DATA=ROOT/'experiments/data_study'
STUDY=ROOT/'experiments/architecture_study'
MODEL=ROOT/'models/optimize/cle_morphology.py'
ORIGINAL_MODEL=core.new_model
core.STUDY=STUDY
core.WEIGHTS=ROOT/'weights/architecture_study'


def sources():
    return dict(driver=core.digest(__file__),model=core.digest(MODEL),
                trainer=core.digest(ROOT/'experiments/run_data_study.py'))


def model_for(seed,variant,position=3,control_width=64):
    from models.optimize.cle_morphology import MorphologyResNet
    return MorphologyResNet(ORIGINAL_MODEL(seed),variant,position,control_width).cuda()


def verify():
    cfg=core.read_json(STUDY/'config.json')
    assert cfg['architecture_sources']==sources(),'Source changed; do not silently reuse earlier artifacts'
    assert core.digest(STUDY/'manifest.json')==cfg['manifest_sha256']
    assert core.digest(DATA/'train_grades.csv')==cfg['grade_manifest_sha256']
    assert core.digest(DATA/'config.json')==cfg['data_protocol_sha256']
    assert core.read_json(core.CACHE/'meta.json')['manifest_sha256']==cfg['manifest_sha256']
    return cfg


def commit(message):
    subprocess.run([core.GIT,'-C',str(ROOT),'add','experiments','models/optimize/cle_morphology.py','ARCHITECTURE_PROTOCOL.md','.gitignore'],check=True)
    if subprocess.run([core.GIT,'-C',str(ROOT),'diff','--cached','--quiet']).returncode:
        subprocess.run([core.GIT,'-C',str(ROOT),'commit','-m',message],check=True)


def prepare():
    import numpy as np
    from models.optimize.cle_morphology import CLEContext,PlainControl
    STUDY.mkdir(parents=True,exist_ok=True)
    if (STUDY/'config.json').exists():verify();return
    cfg=core.read_json(DATA/'config.json')
    target=sum(p.numel() for p in CLEContext(256).parameters())
    candidates={w:sum(p.numel() for p in PlainControl(256,w).parameters()) for w in range(32,257,8)}
    width=min(candidates,key=lambda w:abs(candidates[w]-target))
    cfg.update(architecture_sources=sources(),created_at=core.timestamp(),
               grade_manifest_sha256=core.digest(DATA/'train_grades.csv'),data_protocol_sha256=core.digest(DATA/'config.json'),
               round=1,recipes=['ac_only','grade_weighted'],variants=['conv_control','ls_reference','cle_context'],
               control_width=width,control_parameters=candidates[width],cle_parameters=target,
               parameter_match_relative_error=abs(candidates[width]-target)/target,
               primary_hypothesis='Local contrast plus bounded context aggregation improves development-patient morphology classification beyond plain convolution and LS reference.',
               selection_policy='Development-only paired seeds and patient intervals. No test use. Both recipes retained because data-study validation and test rankings differed.',
               test_policy='This driver NEVER evaluates test; no A-C test filtering.')
    (STUDY/'manifest.json').write_bytes((DATA/'manifest.json').read_bytes())
    original=core.read_json(DATA/'final_jobs.json')
    jobs=[]
    for recipe in cfg['recipes']:
        for variant in cfg['variants']:
            for seed in cfg['seeds']:
                base=next(j for j in original if j['arm']==recipe and j['seed']==seed)
                job=dict(base)
                job.update(id=f'r1_{recipe}_{variant}_s{seed}',arm=f'{recipe}/{variant}',phase='architecture',
                           recipe=recipe,variant=variant,position=3,control_width=width,
                           score=cfg['val_indices'],architecture_sources=cfg['architecture_sources'])
                # Both select and score refer to the development set. These are
                # validation-selected exploratory scores, never claimed as OOF.
                assert set(job['train']).isdisjoint(job['select'])
                assert set(job['score'])==set(cfg['val_indices'])
                jobs.append(job)
    core.save_json(STUDY/'config.json',cfg)
    core.save_json(STUDY/'final_jobs.json',jobs)
    core.save_json(STUDY/'status.json',dict(phase='prepared',updated_at=core.timestamp(),total=len(jobs)))
    commit('research: freeze six development-only architecture/data cells and matched convolution budget')
    print(json.dumps(dict(jobs=len(jobs),control_width=width,control_parameters=candidates[width],cle_parameters=target)),flush=True)


def read_csv(p):
    with open(p) as f:return list(csv.DictReader(f))


def baseline_predictions():
    import torch
    cfg=verify()
    for recipe in cfg['recipes']:
        for seed in cfg['seeds']:
            out=STUDY/'baselines'/f'{recipe}_s{seed}'
            if (out/'validation_result.json').exists():continue
            old=core.read_json(DATA/'runs'/f'{recipe}_s{seed}'/'complete.json')
            ck=ROOT/old['checkpoint']
            assert core.digest(ck)==old['checkpoint_sha256']
            m=ORIGINAL_MODEL(seed)
            m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=True)
            p,y,idx=core.infer(m,core.loader(cfg['val_indices'],False,seed=seed))
            result=core.metrics(p,y)
            assert abs(result['macro_f1']-old['best_select_f1'])<1e-10,'baseline validation reproduction failed'
            core.save_csv(out/'validation_predictions.csv',core.prediction_rows(p,y,idx))
            core.save_json(out/'validation_result.json',dict(recipe=recipe,seed=seed,
                           checkpoint_sha256=old['checkpoint_sha256'],n=len(y),**result))
            del m
            torch.cuda.empty_cache()
    commit('results: reproduce six frozen baseline development predictions without new test evaluation')


def worker(job_id):
    import numpy as np
    import torch
    verify()
    job=next(j for j in core.read_json(STUDY/'final_jobs.json') if j['id']==job_id)
    core.new_model=lambda seed:model_for(seed,job['variant'],job['position'],job['control_width'])
    core.run_worker(job_id)
    out=STUDY/'runs'/job_id
    completion=core.read_json(out/'complete.json')
    model=core.new_model(job['seed'])
    model.load_state_dict(torch.load(ROOT/completion['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    perturb=[];routes=[]
    def hook(module,inputs,output):
        x=inputs[0]
        perturb.extend(((output-x).abs().flatten(1).mean(1)/(x.abs().flatten(1).mean(1)+1e-8)).detach().cpu().float().tolist())
        r=getattr(module.branch,'last_routing',None)
        if r is not None:routes.extend(r[:,0].flatten(1).mean(1).cpu().float().tolist())
    handle=model.modifier.register_forward_hook(hook)
    p,y,idx=core.infer(model,core.loader(job['score'],False,seed=job['seed']))
    handle.remove()
    core.save_csv(out/'validation_predictions.csv',core.prediction_rows(p,y,idx))
    model.modifier.enabled=False
    po,yo,idxo=core.infer(model,core.loader(job['score'],False,seed=job['seed']))
    assert idx==idxo and np.array_equal(y,yo)
    diag=dict(stream_perturbation_mean=float(np.mean(perturb)),
              stream_perturbation_quantiles=np.percentile(perturb,[0,25,50,75,100]).tolist(),
              route_local_mean=float(np.mean(routes)) if routes else None,
              off_same_checkpoint_metrics=core.metrics(po,yo),
              off_prediction_disagreement=float(np.mean(p.argmax(1)!=po.argmax(1))),
              note='Same fitted checkpoint diagnostic, not a substitute for retrained ablation.')
    core.save_json(out/'validation_complete.json',dict(signature=core.signature(job),job_id=job_id,
                   recipe=job['recipe'],variant=job['variant'],seed=job['seed'],n=len(y),
                   checkpoint_sha256=completion['checkpoint_sha256'],completed_at=core.timestamp(),
                   parameters=sum(p.numel() for p in model.parameters()),
                   modifier_parameters=sum(p.numel() for p in model.modifier.parameters()),
                   diagnostics=diag,**core.metrics(p,y)))


def done(job):
    out=STUDY/'runs'/job['id']
    if not (out/'validation_complete.json').exists():return False
    assert core.complete(job)
    m=core.read_json(out/'validation_complete.json')
    assert m['signature']==core.signature(job) and m['n']==538
    return True


def run_jobs(jobs):
    pending=[j for j in jobs if not done(j)];running={};retry=collections.Counter()
    logs=STUDY/'logs';logs.mkdir(exist_ok=True)
    while pending or running:
        while pending and len(running)<2:
            j=pending.pop(0);lf=open(logs/(j['id']+'.log'),'a',buffering=1)
            env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            proc=subprocess.Popen([sys.executable,__file__,'worker','--job',j['id']],cwd=ROOT,
                                  env=env,stdin=subprocess.DEVNULL,stdout=lf,stderr=subprocess.STDOUT)
            running[proc.pid]=(proc,j,lf)
            print('[launch]',j['id'],proc.pid,flush=True)
        core.save_json(STUDY/'status.json',dict(phase='round1_training',updated_at=core.timestamp(),total=len(jobs),
                       completed=len(jobs)-len(pending)-len(running),pending=len(pending),
                       running=[dict(job=j['id'],pid=p.pid) for p,j,_ in running.values()]))
        for pid,(p,j,lf) in list(running.items()):
            if p.poll() is None:continue
            lf.close();del running[pid]
            if p.returncode or not done(j):
                retry[j['id']]+=1
                if retry[j['id']]<=1:
                    pending.append(j);print('[retry]',j['id'],p.returncode,flush=True)
                else:
                    for q,_,f in running.values():q.wait();f.close()
                    raise RuntimeError(j['id']+' failed twice; inspect its log')
            else:
                commit('results: complete development-only architecture run '+j['id'])
                print('[done]',j['id'],flush=True)
        if pending or running:time.sleep(15)


def summarize():
    import numpy as np
    cfg=verify();jobs=core.read_json(STUDY/'final_jobs.json')
    results=[core.read_json(STUDY/'runs'/j['id']/'validation_complete.json') for j in jobs]
    groups=None;weight=None;summary=[]
    def cm(rows):
        nonlocal groups,weight
        if groups is None:
            groups=sorted({r['group_id'] for r in rows})
            weight=np.random.default_rng(20261024).multinomial(len(groups),np.ones(len(groups))/len(groups),size=20000)
        matrix=np.zeros((len(groups),3,3))
        np.add.at(matrix,([groups.index(r['group_id']) for r in rows],[int(r['label']) for r in rows],[int(r['pred']) for r in rows]),1)
        return matrix
    def f1(matrix):
        den=matrix.sum(-1)+matrix.sum(-2)
        return np.mean(np.divide(2*np.diagonal(matrix,axis1=-2,axis2=-1),den,out=np.zeros_like(den),where=den!=0),axis=-1)
    for recipe in cfg['recipes']:
        for variant in cfg['variants']:
            runs=[r for r in results if r['recipe']==recipe and r['variant']==variant]
            deltas=[];boot=[]
            for seed in cfg['seeds']:
                job=next(j for j in jobs if j['recipe']==recipe and j['variant']==variant and j['seed']==seed)
                a=sorted(read_csv(STUDY/'baselines'/f'{recipe}_s{seed}'/'validation_predictions.csv'),key=lambda r:int(r['index']))
                b=sorted(read_csv(STUDY/'runs'/job['id']/'validation_predictions.csv'),key=lambda r:int(r['index']))
                assert [r['index'] for r in a]==[r['index'] for r in b]
                ma,mb=cm(a),cm(b)
                deltas.append(float(f1(mb.sum(0))-f1(ma.sum(0))))
                boot.append(f1(np.einsum('bg,gij->bij',weight,mb))-f1(np.einsum('bg,gij->bij',weight,ma)))
            summary.append(dict(recipe=recipe,variant=variant,mean_validation_f1=float(np.mean([r['macro_f1'] for r in runs])),
                                seed_sd=float(np.std([r['macro_f1'] for r in runs],ddof=1)),seed_deltas=deltas,
                                mean_delta=float(np.mean(deltas)),patient_cluster95=np.percentile(np.mean(boot,axis=0),[2.5,97.5]).tolist(),
                                mean_malignant_f1=float(np.mean([r['class_f1'][1] for r in runs])),
                                modifier_parameters=runs[0]['modifier_parameters']))
    core.save_json(STUDY/'round1_results.json',results)
    core.save_json(STUDY/'round1_summary.json',summary)
    lines=['# CLE architecture round 1: development results','',
           'Exploratory, validation-selected scores. The intervals condition on the fitted pairs and do not correct for model selection or multiple candidates.',
           'No test images were evaluated by this architecture study.','',
           '| Recipe | Variant | Mean val F1 | Delta to paired plain ResNet | Patient interval | Extra parameters |',
           '|---|---|---:|---:|---|---:|']
    for r in summary:
        lines.append(f'| {r["recipe"]} | {r["variant"]} | {r["mean_validation_f1"]:.6f} | {r["mean_delta"]:+.6f} | {r["patient_cluster95"]} | {r["modifier_parameters"]} |')
    lines+=['','Status: requires mechanism analysis and an independently specified confirmation/ablation round, not a publication-success declaration.',
            'The architecture_sources, checkpoint SHA256, full per-image predictions, branch off diagnostics and training logs are preserved.']
    (STUDY/'ROUND1_REPORT.md').write_text('\n'.join(lines))
    commit('results: summarize all architecture cells with development-patient paired uncertainty')
    core.save_json(STUDY/'status.json',dict(phase='round_complete_needs_analysis',updated_at=core.timestamp(),completed=len(jobs),total=len(jobs),
                   report='experiments/architecture_study/ROUND1_REPORT.md'))


def check():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from models.optimize.cle_morphology import shared_aggregate,MorphologyBlock
    cfg=verify();core.set_seed(42)
    x=torch.randn(1,16,5,7,dtype=torch.float64,requires_grad=True)
    w=torch.randn(1,2,9,5,7,dtype=torch.float64,requires_grad=True)
    fast=shared_aggregate(x,w)
    patches=F.unfold(x,3,padding=1).reshape(1,16,9,5,7)
    slow=torch.stack([(patches[:,c]*w[:,c%2]).sum(1) for c in range(16)],1)
    assert torch.allclose(fast,slow,atol=1e-10)
    gf=torch.autograd.grad(fast.square().sum(),(x,w),retain_graph=True)
    gs=torch.autograd.grad(slow.square().sum(),(x,w))
    assert all(torch.allclose(a,b,atol=1e-8) for a,b in zip(gf,gs))
    checked=[]
    for variant in cfg['variants']:
        for hw in [7,14,28]:
            block=MorphologyBlock(256,variant,cfg['control_width']).cuda()
            z=torch.randn(2,256,hw,hw,device='cuda',requires_grad=True)
            with torch.autocast('cuda',dtype=torch.float16):out=block(z);loss=out.square().mean()
            loss.backward()
            assert out.shape==z.shape and torch.isfinite(out).all() and torch.isfinite(z.grad).all()
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in block.parameters())
            checked.append([variant,hw])
    backbone=ORIGINAL_MODEL(42).eval()
    baseline_state={k:v.detach().cpu().clone() for k,v in backbone.state_dict().items()}
    batch=next(iter(core.loader(cfg['val_indices'][:32],False,seed=42)))[0].cuda()
    for variant in cfg['variants']:
        model=model_for(42,variant,control_width=cfg['control_width']).eval()
        assert all(torch.equal(v.cpu(),baseline_state[k]) for k,v in model.backbone.state_dict().items())
        model.modifier.enabled=False
        with torch.inference_mode():assert torch.equal(model(batch),model.backbone(batch))
        model.modifier.enabled=True
        with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model.train()(batch),torch.zeros(32,device='cuda',dtype=torch.long))
        loss.backward();assert torch.isfinite(loss)
        del model
    core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),
                   checks=['official SKA modulo sharing matches independent loop in forward/backward','AMP finite gradients at 7/14/28 spatial sizes',
                           'identical pretrained backbone/head per paired seed','disabled modifier equals backbone exactly','full-model CUDA forward/backward'],cases=checked))
    commit('test: verify reference aggregation, module gradients and paired pretrained initialization')
    print('PASS architecture preflight',flush=True)


def run():
    STUDY.mkdir(parents=True,exist_ok=True)
    with open(STUDY/'pipeline.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            verify()
            assert core.read_json(STUDY/'preflight_validation.json')['passed']
            baseline_predictions()
            run_jobs(core.read_json(STUDY/'final_jobs.json'))
            summarize()
        except Exception as e:
            core.save_json(STUDY/'status.json',dict(phase='failed',updated_at=core.timestamp(),error=repr(e)))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','check','worker','run']);p.add_argument('--job')
    a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='check':check()
    elif a.command=='worker':worker(a.job)
    else:run()
