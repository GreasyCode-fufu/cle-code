"""Run explicitly requested experiments under the validated reference protocol."""
import argparse,csv,json,os,shutil,statistics,subprocess,sys,time
from datetime import datetime
from pathlib import Path
from preflight import check,ROOT

MODELS=('convnext_tiny','convnext_egb','convnext_scg','convnext_hsc','convnext_esa',
        'convnext_egb_scg','convnext_scg_hsc','convnext_egb_scg_hsc','convnext_esa_scg_hsc')
METRICS=('macro_f1','accuracy','weighted_f1','balanced_accuracy','auc_roc_macro',
         'macro_sensitivity','macro_specificity','kappa','mcc')
def now():return datetime.now().astimezone().isoformat()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--check-only',action='store_true')
 p.add_argument('--run',action='store_true',help='Explicitly start GPU training')
 p.add_argument('--models',nargs='+',choices=MODELS,default=['convnext_tiny','convnext_egb_scg'])
 p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
 args=p.parse_args()
 if not args.run and not args.check_only:p.error('Choose --check-only or --run')
 result=check();print(json.dumps(result,indent=2),flush=True)
 if args.check_only:return
 if len(set(args.models))!=len(args.models) or len(set(args.seeds))!=len(args.seeds):p.error('Duplicate model or seed')
 lock=json.loads((ROOT/'protocol_lock.json').read_text())
 output=ROOT/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');output.mkdir(parents=True,exist_ok=False)
 def record(event,**kw):
  row=dict(time=now(),event=event,**kw)
  with (output/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
 for f in ('protocol_lock.json','source_files.sha256.json'):shutil.copy2(ROOT/f,output/f)
 git_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 record('started',git_commit=git_commit,python=sys.executable,models=args.models,seeds=args.seeds)
 def execute(command,log,**info):
  with log.open('w') as f:
   child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   record('process_started',pid=child.pid,command=command,**info)
   rc=child.wait()
  record('process_finished',returncode=rc,**info)
  if rc:raise RuntimeError(f'Command failed ({rc}): {log}')
 for seed in args.seeds:
  for model in args.models:
   while True:
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if not gpu:break
    record('waiting_for_gpu',pids=gpu.splitlines());time.sleep(30)
   base=output/f'{model}_seed{seed}';base.mkdir()
   command=[sys.executable,str(ROOT/'train_cle_fair.py'),'--model',model,'--num_classes','3',
    '--img_size','224','--seed',str(seed),'--num-workers','3','--batch-size','32','--lr','0.0001',
    '--epochs','300','--early-stop','30','--label-smoothing','0','--device','cuda:0',
    '--data-path-train',str(Path(lock['dataset'])/'train'),'--data-path-val',str(Path(lock['dataset'])/'val'),
    '--weights',lock['pretrained'],'--save-dir',str(base)]
   execute(command,base/'train.log',model=model,seed=seed,phase='train')
   run=base/model
   if not (run/'model_best.pth').exists():raise RuntimeError('Missing new checkpoint')
   for split in ('val','test'):
    command=[sys.executable,str(ROOT/'evaluate.py'),'--model',model,'--weights',str(run/'model_best.pth'),
     '--data-path',str(Path(lock['dataset'])/split),'--run-meta',str(run/'run_meta.json'),
     '--img-size','224','--batch-size','64','--device','cuda:0','--out-dir',str(run/f'eval_{split}')]
    execute(command,base/f'eval_{split}.log',model=model,seed=seed,phase=split)
   record('run_complete',model=model,seed=seed)
 rows=[];summary=[]
 for model in args.models:
  for seed in args.seeds:
   f=output/f'{model}_seed{seed}/{model}/eval_test/results.json';d=json.loads(f.read_text())
   if d['n_total']!=576:raise RuntimeError('Test sample count changed')
   rows.append(dict(model=model,seed=seed,source=str(f),**{k:d[k] for k in METRICS}))
  r=[v for v in rows if v['model']==model];item=dict(model=model,n=len(r))
  for k in METRICS:
   item[k+'_mean']=statistics.mean(v[k] for v in r)
   item[k+'_sample_std']=statistics.stdev(v[k] for v in r) if len(r)>1 else None
  summary.append(item)
 for name,data in [('per_seed.csv',rows),('summary.csv',summary)]:
  with (output/name).open('w',newline='',encoding='utf-8-sig') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 (output/'summary.json').write_text(json.dumps(summary,indent=2))
 record('all_complete',summary=summary)

if __name__=='__main__':main()
