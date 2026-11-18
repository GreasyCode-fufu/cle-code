"""Independent paired rerun from the audited historical Git snapshot.

Never reads checkpoints from other experiments. Every run starts from ImageNet.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time
from datetime import datetime

SOURCE = Path('/home/login/data/cle/auto/Network/CLE-Net')
OUT = SOURCE / 'historical_protocol_codex_20261109'
DATA = SOURCE / 'datasets/cle_ac_v1'
PRETRAINED = SOURCE / 'pretrained/convnext_tiny_1k_224_ema.pth'
REV = 'b457d93'
ENV = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
METRICS = ['macro_f1', 'accuracy', 'weighted_f1', 'balanced_accuracy', 'auc_roc_macro',
           'macro_sensitivity', 'macro_specificity', 'kappa', 'mcc']

def now():
    return datetime.now().astimezone().isoformat()

def record(event, **extra):
    row = dict(time=now(), event=event, **extra)
    with (OUT / 'events.jsonl').open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def wait_gpu():
    notified = False
    while True:
        p = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                           capture_output=True, text=True, check=True)
        if not p.stdout.strip():
            return
        if not notified:
            record('waiting_for_other_gpu_jobs', pids=p.stdout.strip().splitlines())
            notified=True
        time.sleep(30)

def execute(command, log, **extra):
    with log.open('w') as f:
        f.write(json.dumps(command) + '\n'); f.flush()
        p = subprocess.Popen(command, cwd=OUT/'code', env=ENV,
                             stdout=f, stderr=subprocess.STDOUT)
        record('process_started', pid=p.pid, command=command, log=str(log), **extra)
        rc = p.wait()
    record('process_finished', returncode=rc, **extra)
    if rc:
        raise RuntimeError(f'Failed: {log}, returncode={rc}')

def aggregate():
    rows=[]
    for seed in (42,43,44):
        for model in ('convnext_tiny','convnext_egb_scg'):
            run=OUT/'runs'/f'{model}_seed{seed}'/model
            d=json.loads((run/'eval_test/results.json').read_text())
            m=json.loads((run/'run_meta.json').read_text())
            assert d['n_total']==576
            assert m['pretrained_load']['checkpoint_sha256']==sha(PRETRAINED)
            logs=list(csv.DictReader((run/'train_log.csv').open()))
            best=max(logs,key=lambda r:float(r['val_macro_f1']))
            rows.append(dict(model=model,seed=seed,best_epoch=int(best['epoch']),
                             stop_epoch=int(logs[-1]['epoch']),
                             val_macro_f1=float(best['val_macro_f1']),
                             source=str(run/'eval_test/results.json'),
                             **{k:d[k] for k in METRICS}))
    with (OUT/'per_seed.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for model in ('convnext_tiny','convnext_egb_scg'):
        rr=[r for r in rows if r['model']==model]
        s=dict(model=model,n_seeds=3)
        for k in METRICS:
            s[k+'_mean']=statistics.mean(r[k] for r in rr)
            s[k+'_sample_std']=statistics.stdev(r[k] for r in rr)
        summary.append(s)
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    paired=[dict(seed=s,delta_f1=next(r['macro_f1'] for r in rows if r['seed']==s and r['model']=='convnext_egb_scg')-
                 next(r['macro_f1'] for r in rows if r['seed']==s and r['model']=='convnext_tiny')) for s in (42,43,44)]
    (OUT/'paired_differences.json').write_text(json.dumps(paired,indent=2))
    return summary

def main():
    OUT.mkdir(exist_ok=False)
    (OUT/'logs').mkdir();(OUT/'code').mkdir();(OUT/'runs').mkdir()
    commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse',REV],text=True).strip()
    record('setup_started',pid=os.getpid(),python=sys.executable,commit=commit)
    archive=OUT/'historical_source.tar'
    with archive.open('wb') as f:
        subprocess.run(['git','-C',str(SOURCE),'archive',commit],stdout=f,check=True)
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            if member.isfile() or member.isdir():
                target=(OUT/'code'/member.name).resolve()
                assert target.is_relative_to((OUT/'code').resolve())
                t.extract(member,OUT/'code',filter='data')
    probe=subprocess.check_output([sys.executable,'-c',
        'import sys,torch,torchvision;print(sys.version);print(torch.__version__);print(torchvision.__version__);print(torch.version.cuda)'],text=True,env=ENV)
    (OUT/'environment_versions.txt').write_text(probe)
    with (OUT/'pip_freeze.txt').open('w') as f:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,env=ENV,check=True)
    (OUT/'nvidia_smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    manifest=[]
    for split in ('train','val','test'):
        for p in sorted((DATA/split).rglob('*')):
            if p.is_file() and p.suffix.lower() in ('.jpg','.png','.jpeg','.tif','.tiff','.bmp'):
                manifest.append(dict(split=split,path=str(p.relative_to(DATA/split)),sha256=sha(p)))
    (OUT/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
    protocol=dict(commit=commit,source=str(SOURCE),data=str(DATA),data_resolved=str(DATA.resolve()),
                  pretrained=str(PRETRAINED),pretrained_sha256=sha(PRETRAINED),
                  source_archive_sha256=sha(archive),python=sys.executable,
                  batch_size=32,lr=1e-4,gate_lr=1e-2,weight_decay=0.05,gate_weight_decay=0,
                  epochs=300,patience=30,workers=3,omp_threads=2,mkl_threads=2,
                  label_smoothing=0,selection='validation_macro_f1',seeds=[42,43,44],
                  historical_environment_fully_recovered=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    record('setup_complete',versions=probe)
    for seed in (42,43,44):
        for model in ('convnext_tiny','convnext_egb_scg'):
            wait_gpu()
            save=OUT/'runs'/f'{model}_seed{seed}'
            command=[sys.executable,str(OUT/'code/train_cle_fair.py'),
                '--model',model,'--seed',str(seed),'--num_classes','3','--img_size','224',
                '--batch-size','32','--lr','0.0001','--epochs','300','--early-stop','30',
                '--num-workers','3','--label-smoothing','0',
                '--data-path-train',str(DATA/'train'),'--data-path-val',str(DATA/'val'),
                '--weights',str(PRETRAINED),'--save-dir',str(save),'--device','cuda:0']
            execute(command,OUT/'logs'/f'{model}_seed{seed}_train.log',model=model,seed=seed,phase='train')
            run=save/model
            assert (run/'model_best.pth').is_file()
            for split in ('val','test'):
                command=[sys.executable,str(OUT/'code/evaluate.py'),'--model',model,
                    '--weights',str(run/'model_best.pth'),'--data-path',str(DATA/split),
                    '--run-meta',str(run/'run_meta.json'),'--img-size','224',
                    '--batch-size','64','--device','cuda:0','--out-dir',str(run/f'eval_{split}')]
                execute(command,OUT/'logs'/f'{model}_seed{seed}_{split}.log',model=model,seed=seed,phase=split)
            record('run_complete',model=model,seed=seed)
    summary=aggregate()
    record('all_complete',summary=summary)

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        if (OUT/'events.jsonl').exists():record('failed',error=repr(e))
        raise
