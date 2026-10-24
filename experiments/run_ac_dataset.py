#!/usr/bin/env python3
"""New AC-only cohort task: split-local grading, frozen dataset, then baseline."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import run_architecture_study as arch
core=arch.core;DATA=ROOT/'experiments/data_study';STUDY=ROOT/'experiments/ac_dataset_v1'
MODEL_STUDY=ROOT/'experiments/ac_models_v1';DATASET=ROOT/'datasets/cle_ac_v1'
ORIGINAL_MODEL=arch.ORIGINAL_MODEL
core.STUDY=STUDY;core.WEIGHTS=ROOT/'weights/ac_dataset_v1'
SEEDS=[42,43,44]


def new_model(seed):
    import torch
    model=ORIGINAL_MODEL(seed);torch.set_num_threads(1)
    return model


core.new_model=new_model


def commit(message):
    subprocess.run([core.GIT,'-C',str(ROOT),'add','experiments/ac_dataset_v1','experiments/ac_models_v1',
                    'experiments/run_ac_dataset.py','experiments/AC_DATASET_V1_PROTOCOL.md','experiments/RESEARCH_LEDGER.md','.gitignore'],check=True)
    if subprocess.run([core.GIT,'-C',str(ROOT),'diff','--cached','--quiet']).returncode:
        subprocess.run([core.GIT,'-C',str(ROOT),'commit','-m',message],check=True)


def sources():
    import sklearn,torch
    return dict(driver=core.digest(__file__),protocol=core.digest(ROOT/'experiments/AC_DATASET_V1_PROTOCOL.md'),
                trainer=core.digest(core.__file__),supervisor=core.digest(ROOT/'experiments/run_architecture_study.py'),
                backbone=core.digest(ROOT/'cle_fair_common.py'),sklearn=sklearn.__version__,torch=torch.__version__)


def old_signature(job):
    return hashlib.sha256((json.dumps(job,sort_keys=True)+core.digest(DATA/'config.json')+core.digest(core.__file__)).encode()).hexdigest()


def verify():
    cfg=core.read_json(STUDY/'config.json');assert cfg['sources']==sources()
    for path,key in [(STUDY/'manifest.json','manifest_sha256'),(STUDY/'final_jobs.json','jobs_sha256'),
                     (DATA/'config.json','source_data_config_sha256'),(DATA/'oof_jobs.json','source_oof_jobs_sha256'),
                     (core.CACHE/'meta.json','cache_meta_sha256')]:assert core.digest(path)==cfg[key]
    for name,sha in cfg['reused_file_hashes'].items():assert core.digest(STUDY/name)==sha
    return cfg


def prepare():
    from sklearn.model_selection import StratifiedGroupKFold
    STUDY.mkdir(parents=True,exist_ok=True);MODEL_STUDY.mkdir(parents=True,exist_ok=True)
    if not (MODEL_STUDY/'README.md').exists():
        (MODEL_STUDY/'README.md').write_text('AC-only model study. Cohort lists must freeze before baseline/architecture jobs. Test is fixed AC subset.\n')
    if (STUDY/'config.json').exists():verify();return
    (STUDY/'manifest.json').write_bytes((DATA/'manifest.json').read_bytes())
    records=core.read_json(STUDY/'manifest.json');jobs=[];reused=[];hashes={}
    for old in core.read_json(DATA/'oof_jobs.json'):
        comp=core.read_json(DATA/'runs'/old['id']/'complete.json')
        assert comp['signature']==old_signature(old) and comp['epochs_completed']==60
        assert core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']
        out=STUDY/'reused_train'/old['id'];out.mkdir(parents=True,exist_ok=True)
        for name in ['complete.json','run_meta.json','train_log.csv','oof_predictions.csv']:
            (out/name).write_bytes((DATA/'runs'/old['id']/name).read_bytes());hashes[str((out/name).relative_to(STUDY))]=core.digest(out/name)
        reused.append(dict(source_job=old,checkpoint_sha256=comp['checkpoint_sha256'],source_signature=comp['signature']))
    core.save_json(STUDY/'reused_index.json',reused);hashes['reused_index.json']=core.digest(STUDY/'reused_index.json')
    for split in ['val','test']:
        ids=np.array([r['index'] for r in records if r['split']==split]);y=np.array([records[i]['label'] for i in ids]);groups=np.array([records[i]['group_id'] for i in ids])
        for seed in SEEDS:
            for fold,(a,b) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(ids,y,groups)):
                rest,score=ids[a],ids[b];chosen=None
                for c,d in StratifiedGroupKFold(10,shuffle=True,random_state=seed*100+fold).split(rest,y[a],groups[a]):
                    if all({records[int(i)]['label'] for i in z}=={0,1,2} for z in [rest[c],rest[d],score]):chosen=(rest[c],rest[d]);break
                assert chosen is not None
                fit,select=chosen;core.validate_split(records,fit,select,score)
                jobs.append(dict(id=f'acv1_grade_{split}_s{seed}_f{fold}',phase='oof',arm='split_local_grader',grade_split=split,
                                 seed=seed,fold=fold,train=fit.tolist(),select=select.tolist(),score=score.tolist(),epochs=60,sample_weights={}))
    core.save_json(STUDY/'final_jobs.json',jobs)
    cfg=dict(task='AC_only_dataset_v1',created_at=core.timestamp(),sources=sources(),manifest_sha256=core.digest(STUDY/'manifest.json'),
             jobs_sha256=core.digest(STUDY/'final_jobs.json'),source_data_config_sha256=core.digest(DATA/'config.json'),
             source_oof_jobs_sha256=core.digest(DATA/'oof_jobs.json'),cache_meta_sha256=core.digest(core.CACHE/'meta.json'),
             pretrained_sha256=core.digest(core.PRETRAINED),seeds=SEEDS,epochs=60,new_grade_models=30,reused_grade_models=15,
             max_parallel=6,cpu_threads=1,loader_workers=2,reserve_gpu_mib=6144,new_job_gpu_mib=2048,
             grading='split-local patient5fold; inner10 first-valid-three-class; original core.grade thresholds',
             objective='Optimize/design networks only on retained AC train/val/test; shared frozen subset.',reused_file_hashes=hashes)
    core.save_json(STUDY/'config.json',cfg);core.save_json(STUDY/'status.json',dict(phase='prepared',total=30,completed=0))
    commit('data: freeze new split-local AC cohort grading and reuse15 signed train models')


def pred_metrics(path,indices,records):
    rows=arch.read_csv(path);assert [int(r['index']) for r in rows]==indices
    p=np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows]);y=np.array([records[i]['label'] for i in indices])
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=2e-6)
    assert np.array_equal(p.argmax(1),[int(r['pred']) for r in rows])
    for r,i in zip(rows,indices):assert r['group_id']==records[i]['group_id'] and int(r['label'])==records[i]['label']
    return rows,core.metrics(p,y)


def grade_done(job):
    path=STUDY/'runs'/job['id']/'grade_complete.json'
    if not path.exists():return False
    assert core.complete(job)
    value=core.read_json(path);assert value['signature']==core.signature(job)
    assert core.digest(path.parent/'oof_predictions.csv')==value['prediction_sha256']
    return True


def activate_baseline():
    core.STUDY=MODEL_STUDY;core.WEIGHTS=ROOT/'weights/ac_models_v1'


def verify_models():
    cfg=core.read_json(MODEL_STUDY/'config.json');assert cfg['sources']==sources()
    assert core.digest(STUDY/'dataset_config.json')==cfg['cohort_config_sha256']
    assert core.digest(MODEL_STUDY/'manifest.json')==cfg['manifest_sha256']
    assert core.digest(MODEL_STUDY/'final_jobs.json')==cfg['jobs_sha256']
    dataset=core.read_json(STUDY/'dataset_config.json')
    assert core.digest(STUDY/'dataset_manifest.json')==dataset['manifest_sha256']
    for split,sha in dataset['grade_csv_sha256'].items():assert core.digest(STUDY/f'{split}_grades.csv')==sha
    return cfg


def baseline_done(job):
    path=MODEL_STUDY/'runs'/job['id']/'validation_complete.json'
    if not path.exists():return False
    value=core.read_json(path)
    assert core.complete(job) and value['signature']==core.signature(job)
    assert core.digest(path.parent/'validation_predictions.csv')==value['prediction_sha256']
    return True


def worker(job_id):
    import torch
    verify()
    if job_id.startswith('acv1_base_'):
        verify_models();activate_baseline();job=next(j for j in core.read_json(MODEL_STUDY/'final_jobs.json') if j['id']==job_id)
        if baseline_done(job):return
        if not core.complete(job):core.run_worker(job_id)
        comp=core.read_json(MODEL_STUDY/'runs'/job_id/'complete.json');m=new_model(job['seed'])
        m.load_state_dict(torch.load(ROOT/comp['checkpoint'],map_location='cpu',weights_only=True),strict=True)
        p,y,ids=core.infer(m,core.loader(job['select'],False,seed=job['seed']));metrics=core.metrics(p,y)
        assert abs(metrics['macro_f1']-comp['best_select_f1'])<1e-12
        out=MODEL_STUDY/'runs'/job_id;core.save_csv(out/'validation_predictions.csv',core.prediction_rows(p,y,ids))
        core.save_json(out/'validation_complete.json',dict(signature=core.signature(job),n=len(y),prediction_sha256=core.digest(out/'validation_predictions.csv'),**metrics))
        return
    job=next(j for j in core.read_json(STUDY/'final_jobs.json') if j['id']==job_id)
    if grade_done(job):return
    if not core.complete(job):core.run_worker(job_id)
    out=STUDY/'runs'/job_id;comp=core.read_json(out/'complete.json');records=core.read_json(STUDY/'manifest.json')
    rows,metrics=pred_metrics(out/'oof_predictions.csv',job['score'],records)
    assert abs(metrics['macro_f1']-comp['outer_metrics']['macro_f1'])<1e-12
    core.save_json(out/'grade_complete.json',dict(signature=core.signature(job),n=len(rows),prediction_sha256=core.digest(out/'oof_predictions.csv'),**metrics))


def resources():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]
    util,used,free=[int(v.strip()) for v in raw.split(',')]
    return dict(timestamp=core.timestamp(),gpu_util=util,gpu_used_mib=used,gpu_free_mib=free,cpu_load1=os.getloadavg()[0],cpu_count=os.cpu_count())


def run_jobs(jobs,done,phase):
    cfg=verify();pending=[j for j in jobs if not done(j)];running={};retries=collections.Counter();logs=STUDY/'logs';logs.mkdir(exist_ok=True)
    sample=None;next_sample=0
    while pending or running:
        completed_now=[]
        for pid,(proc,job,handle) in list(running.items()):
            if proc.poll() is None:continue
            handle.close();del running[pid]
            if proc.returncode or not done(job):
                retries[job['id']]+=1
                if retries[job['id']]<=1:pending.append(job);print('[retry]',job['id'],proc.returncode,flush=True)
                else:raise RuntimeError(job['id']+' failed twice; inspect own job log before restarting')
            else:completed_now.append(job['id']);print('[done]',job['id'],flush=True)
        if time.monotonic()>=next_sample:
            sample=resources();sample.update(phase=phase,running=len(running),pending=len(pending))
            path=STUDY/'resource_samples.csv'
            import csv
            exists=path.exists()
            with path.open('a',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(sample));
                if not exists:writer.writeheader()
                writer.writerow(sample)
            next_sample=time.monotonic()+2
        while pending and len(running)<cfg['max_parallel']:
            available=resources()
            if available['gpu_free_mib']<cfg['reserve_gpu_mib']+cfg['new_job_gpu_mib'] or (running and available['cpu_load1']>os.cpu_count()*.9):break
            job=pending.pop(0);f=open(logs/(job['id']+'.log'),'a',buffering=1)
            env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            proc=subprocess.Popen([sys.executable,__file__,'worker','--job',job['id']],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            running[proc.pid]=(proc,job,f);print('[launch]',job['id'],proc.pid,flush=True)
        if completed_now:commit('results: complete AC task jobs '+','.join(completed_now))
        core.save_json(STUDY/'status.json',dict(phase=phase,total=len(jobs),completed=len(jobs)-len(pending)-len(running),pending=len(pending),
                       running=[dict(job=j['id'],pid=p.pid) for p,j,_ in running.values()],updated_at=core.timestamp()))
        if pending or running:time.sleep(.5)


def check():
    import torch
    import torch.nn.functional as F
    cfg=verify();records=core.read_json(STUDY/'manifest.json');jobs=core.read_json(STUDY/'final_jobs.json')
    assert len(jobs)==30 and core.digest(core.CACHE/'resized_uint8.npy')==core.read_json(core.CACHE/'meta.json')['cache_sha256']
    coverage=collections.Counter()
    for job in jobs:
        core.validate_split(records,job['train'],job['select'],job['score'])
        assert {records[i]['split'] for k in ['train','select','score'] for i in job[k]}=={job['grade_split']}
        coverage.update((job['grade_split'],job['seed'],i) for i in job['score'])
    assert set(coverage.values())=={1} and len(coverage)==3*(538+671)
    for entry in core.read_json(STUDY/'reused_index.json'):
        old=entry['source_job'];out=STUDY/'reused_train'/old['id'];comp=core.read_json(out/'complete.json')
        assert comp['signature']==old_signature(old) and core.digest(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']
        assert [int(r['epoch']) for r in arch.read_csv(out/'train_log.csv')]==list(range(60))
        _,m=pred_metrics(out/'oof_predictions.csv',old['score'],records);assert abs(m['macro_f1']-comp['outer_metrics']['macro_f1'])<1e-12
    original=ORIGINAL_MODEL(42).eval();optimized=new_model(42).eval();assert all(torch.equal(v,optimized.state_dict()[k]) for k,v in original.state_dict().items())
    x,y,_,_=next(iter(core.loader(jobs[0]['train'][:32],False,seed=42)));x,y=x.cuda(),y.cuda()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):assert torch.equal(original(x),optimized(x))
    del original;torch.cuda.reset_peak_memory_stats();optimized.train();opt=torch.optim.AdamW(optimized.parameters(),lr=1e-4,weight_decay=.05);scaler=torch.amp.GradScaler('cuda',init_scale=128)
    loader=iter(core.loader([r['index'] for r in records if r['split']=='train'],True,seed=42));wait=step=0.
    for k in range(12):
        start=time.perf_counter();xx,yy,_,_=next(loader);wait+=time.perf_counter()-start
        torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(optimized(xx.cuda(non_blocking=True)),yy.cuda(non_blocking=True))
        scaler.scale(loss).backward();scaler.unscale_(opt)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in optimized.parameters())
        scaler.step(opt);scaler.update();torch.cuda.synchronize();step+=time.perf_counter()-start
    profile=dict(loader_wait_seconds=wait,compute_seconds=step,batches=12,peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                 cpu_threads=torch.get_num_threads(),loader_workers=2,max_parallel=6,resource_snapshot=resources())
    core.save_json(STUDY/'preflight_validation.json',dict(passed=True,timestamp=core.timestamp(),profile=profile,
       checks=['15 reused train checkpoints/signatures/60ep/outer probabilities','30 split-local val/test patient-isolated tasks with exact coverage',
               'full cache SHA and original labels/splits','CPU-thread1 model state and AMP logits identical to reference',
               '12 real training batches finite AMP updates,loader wait and GPU memory profile','original loader2 workers preserved; no augmentation rewrite']))
    commit('test: validate AC dataset grading and GPU CPU loading profile')
    print(json.dumps(profile),flush=True)


def build_dataset():
    cfg=verify();records=core.read_json(STUDY/'manifest.json');pred=collections.defaultdict(list)
    for entry in core.read_json(STUDY/'reused_index.json'):
        job=entry['source_job'];rows,_=pred_metrics(STUDY/'reused_train'/job['id']/'oof_predictions.csv',job['score'],records)
        for row in rows:pred[int(row['index'])].append((job['seed'],np.array([float(row[f'p{k}']) for k in range(3)])))
    for job in core.read_json(STUDY/'final_jobs.json'):
        assert grade_done(job)
        rows,_=pred_metrics(STUDY/'runs'/job['id']/'oof_predictions.csv',job['score'],records)
        for row in rows:pred[int(row['index'])].append((job['seed'],np.array([float(row[f'p{k}']) for k in range(3)])))
    rows=[];retained=[];excluded=[];counts={}
    for record in records:
        i=record['index'];parts=sorted(pred[i],key=lambda t:t[0]);assert [p[0] for p in parts]==SEEDS
        p=np.array([v[1] for v in parts]);correct=float((p.argmax(1)==record['label']).mean());confidence=float(p[:,record['label']].mean())
        grade=core.grade(correct,confidence);row=dict(record,grade=grade,correct_rate=correct,confidence=confidence)
        assert core.digest(ROOT/record['path'])==record['sha256'];rows.append(row)
        if grade in ['A','B','C']:
            link=DATASET/record['split']/record['class_name']/Path(record['path']).name;link.parent.mkdir(parents=True,exist_ok=True)
            if link.is_symlink() or link.exists():assert link.resolve()==(ROOT/record['path']).resolve()
            else:link.symlink_to((ROOT/record['path']).resolve())
            retained.append(dict(row,ac_path=str(link.relative_to(ROOT))))
        else:excluded.append(row)
    assert len(rows)==3595
    original_grades={int(r['index']):r['grade'] for r in arch.read_csv(DATA/'train_grades.csv')}
    assert all(r['grade']==original_grades[r['index']] for r in rows if r['split']=='train')
    for split in ['train','val','test']:
        active=[r for r in retained if r['split']==split];assert {r['label'] for r in active}=={0,1,2}
        core.save_csv(STUDY/f'{split}_grades.csv',[r for r in rows if r['split']==split])
        counts[split]=dict(original=sum(r['split']==split for r in records),retained=len(active),patients=len({r['group_id'] for r in active}),
          grades=dict(collections.Counter(r['grade'] for r in rows if r['split']==split)),class_retained={str(c):sum(r['label']==c for r in active) for c in range(3)})
    core.save_json(STUDY/'dataset_manifest.json',retained);core.save_csv(STUDY/'excluded_DE.csv',excluded)
    dataset_cfg=dict(created_at=core.timestamp(),grading_config_sha256=core.digest(STUDY/'config.json'),manifest_sha256=core.digest(STUDY/'dataset_manifest.json'),
       grade_csv_sha256={split:core.digest(STUDY/f'{split}_grades.csv') for split in ['train','val','test']},counts=counts,
       indices={split:[r['index'] for r in retained if r['split']==split] for split in ['train','val','test']},
       directory=str(DATASET.relative_to(ROOT)),objective='All model optimization and final evaluation restricted to this frozen AC cohort.')
    core.save_json(STUDY/'dataset_config.json',dataset_cfg)
    core.save_json(STUDY/'dataset_audit.json',dict(passed=True,timestamp=core.timestamp(),original_images_sha_verified=3595,
        reused_train_models=15,new_val_test_models=30,predictions_per_image=3,labels_unchanged=True,patient_splits_unchanged=True,counts=counts))
    commit('data: freeze AC train val test lists and materialize shared dataset links')
    print('AC DATASET',json.dumps(counts),flush=True)
    return dataset_cfg


def prepare_baselines(dataset):
    (MODEL_STUDY/'manifest.json').write_bytes((STUDY/'manifest.json').read_bytes());jobs=[]
    for seed in SEEDS:jobs.append(dict(id=f'acv1_base_resnet34_s{seed}',phase='ac_baseline',arm='resnet34',seed=seed,epochs=60,
        train=dataset['indices']['train'],select=dataset['indices']['val'],score=dataset['indices']['val'],sample_weights={}))
    core.save_json(MODEL_STUDY/'final_jobs.json',jobs)
    core.save_json(MODEL_STUDY/'config.json',dict(task='AC_only_model_baselines_v1',sources=sources(),
        cohort_config_sha256=core.digest(STUDY/'dataset_config.json'),pretrained_sha256=core.digest(core.PRETRAINED),
        manifest_sha256=core.digest(MODEL_STUDY/'manifest.json'),jobs_sha256=core.digest(MODEL_STUDY/'final_jobs.json'),
        train_indices=dataset['indices']['train'],val_indices=dataset['indices']['val'],test_indices=dataset['indices']['test'],
        test_policy='Fixed AC test reserved until final candidate frozen; no full-cohort primary override.'))
    commit('research: freeze three ResNet34 baselines on shared AC train and AC val')
    return jobs


def run():
    with open(STUDY/'pipeline.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            verify();assert core.read_json(STUDY/'preflight_validation.json')['passed']
            run_jobs(core.read_json(STUDY/'final_jobs.json'),grade_done,'split_local_grading')
            dataset=core.read_json(STUDY/'dataset_config.json') if (STUDY/'dataset_config.json').exists() else build_dataset()
            jobs=core.read_json(MODEL_STUDY/'final_jobs.json') if (MODEL_STUDY/'config.json').exists() else prepare_baselines(dataset)
            verify_models()
            activate_baseline();run_jobs(jobs,baseline_done,'AC_baseline_training')
            results=[dict(seed=j['seed'],**core.read_json(MODEL_STUDY/'runs'/j['id']/'validation_complete.json')) for j in jobs]
            core.save_json(MODEL_STUDY/'baseline_results.json',results);commit('results: complete three AC-only ResNet34 baselines')
            core.save_json(STUDY/'status.json',dict(phase='AC_dataset_and_baselines_complete_needs_design',counts=dataset['counts'],updated_at=core.timestamp()))
        except Exception as error:
            core.save_json(STUDY/'status.json',dict(phase='failed',error=repr(error),updated_at=core.timestamp()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','check','worker','run']);parser.add_argument('--job');args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='check':check()
    elif args.command=='worker':worker(args.job)
    else:run()
