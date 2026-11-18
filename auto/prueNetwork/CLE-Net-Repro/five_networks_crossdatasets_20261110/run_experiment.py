import csv,hashlib,json,os,shutil,statistics,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path('/home/login/data/cle/auto/prueNetwork/CLE-Net')
ROOT=Path('/home/login/data/cle/auto/prueNetwork/CLE-Net-Repro/five_networks_crossdatasets_20261110')
PY='/home/login/miniconda3/envs/cle_topnets/bin/python'
DATA=Path('/home/login/data/cle/auto/ZCode/Classification/datasets')
MODELS={'mambaout_tiny':'mambaout_tiny.normalized.pth','transnext_tiny':'transnext_tiny_224_1k.pth','mambavision_t':'mambavision_tiny_1k.normalized.pth','thinkingvit_deit':'thinkingvit_deit_3h_6h_imagenet1k.pth','cpubone_b2':'cpubone_b2_bfrobust_r224_in1k.normalized.pth'}
METRICS=['macro_f1','accuracy','weighted_f1','balanced_accuracy','auc_roc_macro','macro_sensitivity','macro_specificity','kappa','mcc']
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
def status(**kw):
 d=dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw);(ROOT/'status.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
def main():
 ROOT.mkdir(exist_ok=True)
 code=ROOT/'code'
 if not code.exists():
  code.mkdir()
  for f in ['train_cle_fair.py','evaluate.py','cle_fair_common.py','utils.py','my_dataset.py','train_utils.py','consensus_manifest.py']:shutil.copy2(BASE/f,code/f)
  shutil.copytree(BASE/'models',code/'models',ignore=shutil.ignore_patterns('__pycache__'))
  shutil.copytree(BASE/'latest_top_networks',code/'latest_top_networks',ignore=shutil.ignore_patterns('results','smoke','__pycache__','.git','*.tgz'))
 specs=[];manifest=[]
 for name,folder,expected in [('BUSI','busi_v1',(546,118,116)),('CUB','cub_v1',(5194,800,5794))]:
  dataset=DATA/folder;classes=sorted(p.name for p in (dataset/'train').iterdir() if p.is_dir());counts=[]
  for split in ['train','val','test']:
   files=sorted(p for p in (dataset/split).rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.tif','.tiff'])
   assert not any('mask' in p.stem.lower() for p in files)
   counts.append(len(files))
   for p in files:manifest.append(dict(dataset=name,split=split,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  assert tuple(counts)==expected,(name,counts,expected)
  specs.append(dict(name=name,path=str(dataset),classes=classes,counts=counts))
 (ROOT/'dataset_manifest.json').write_text(json.dumps(manifest))
 (ROOT/'protocol.json').write_text(json.dumps(dict(datasets=specs,models=MODELS,seeds=[42,43,44],python=PY,epochs=300,patience=30,batch_size=32,lr=.0001,wd=.05,selection='validation Macro-F1',scope='Independent training on each dataset, not zero-shot external transfer'),indent=2))
 (ROOT/'README_AGENT.md').write_text('Five Codex networks, BUSI and CUB only, seeds42/43/44 =30 runs. Covid was abandoned by explicit user request; its retained partial runs are excluded from all final summaries. No CLE-Net/baseline added. Frozen code under code/. Environment cle_topnets. Original ZCode data read-only. MobileNet transfer remains stopped. Inspect status.json and coordinator_busi_cub.log; never treat partial results as completed three-seed means. Each dataset trained independently, not CLE-trained zero-shot transfer. See protocol.json, source_sha256.json, environment.json and SCOPE_CHANGE.json.\n')
 def execute(cmd,log,**info):
  while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():status(phase='waiting_gpu',**info);time.sleep(30)
  with log.open('w') as f:
   p=subprocess.Popen(cmd,cwd=code,stdout=f,stderr=subprocess.STDOUT);status(phase='running',pid=p.pid,command=cmd,**info);rc=p.wait()
  if rc:raise RuntimeError(f'Failed {rc}: {log}')
 allrows=[]
 for spec in specs:
  for seed in [42,43,44]:
   for model,weight in MODELS.items():
    base=ROOT/'runs'/spec['name']/f'{model}_seed{seed}';base.mkdir(parents=True,exist_ok=True);run=base/model
    if not (run/'eval_test/results.json').exists():
     if not (base/'TRAIN_DONE').exists():
      assert not (run/'model_best.pth').exists(),'Partial training requires explicit recovery, not silent restart'
      cmd=[PY,str(code/'latest_top_networks/run_train.py'),'--model',model,'--num_classes',str(len(spec['classes'])),'--seed',str(seed),'--img_size','224','--epochs','300','--early-stop','30','--batch-size','32','--num-workers','3','--lr','0.0001','--label-smoothing','0','--data-path-train',spec['path']+'/train','--data-path-val',spec['path']+'/val','--weights',str(code/'latest_top_networks/pretrained'/weight),'--save-dir',str(base)]
      execute(cmd,base/'train.log',dataset=spec['name'],model=model,seed=seed,stage='train');(base/'TRAIN_DONE').write_text('Normal trainer exit')
     cmd=[PY,str(code/'latest_top_networks/run_eval.py'),'--model',model,'--weights',str(run/'model_best.pth'),'--data-path',spec['path']+'/test','--run-meta',str(run/'run_meta.json'),'--out-dir',str(run/'eval_test'),'--batch-size','64','--class-names',*spec['classes']]
     execute(cmd,base/'test.log',dataset=spec['name'],model=model,seed=seed,stage='test')
    d=json.loads((run/'eval_test/results.json').read_text());assert d['n_total']==spec['counts'][2]
    allrows.append(dict(dataset=spec['name'],model=model,seed=seed,**{m:d[m] for m in METRICS}))
    summary=[]
    for ds,mo in sorted(set((r['dataset'],r['model']) for r in allrows)):
     rr=[r for r in allrows if (r['dataset'],r['model'])==(ds,mo)];s=dict(dataset=ds,model=mo,n=len(rr))
     for m in METRICS:s[m+'_mean']=statistics.mean(r[m] for r in rr);s[m+'_sd']=statistics.stdev(r[m] for r in rr) if len(rr)>1 else None
     summary.append(s)
    for fname,rr in [('per_seed.csv',allrows),('summary.csv',summary)]:
     with (ROOT/fname).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));status(phase='run_complete',completed=len(allrows),total=30,dataset=spec['name'],model=model,seed=seed)
 status(phase='all_complete',completed=30)
if __name__=='__main__':
 try:main()
 except Exception:status(phase='failed',traceback=traceback.format_exc());raise
