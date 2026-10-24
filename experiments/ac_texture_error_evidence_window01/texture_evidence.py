from pathlib import Path
import csv,hashlib,json,os,subprocess,time
S=Path(__file__).resolve().parent;R=S.parents[1];GIT='/home/login/miniconda3/bin/git'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while raw:=f.read(2**20):h.update(raw)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def stamp():
 import datetime
 return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def commit(message):
 while True:
  staged=subprocess.check_output([GIT,'-C',str(R),'diff','--cached','--name-only'],text=True).splitlines();quiet=all(p.startswith(str(S.relative_to(R))+'/') for p in staged)
  c=R/'experiments/ac_crossscale_structure_window01';identity=read(c/'manager_entry_identity_v2.json');proc=Path('/proc')/str(identity['pid'])/'cmdline'
  if proc.exists() and b'run_crossscale_manager.py' in proc.read_bytes():
   st=read(c/'status.json');quiet &= bool(st.get('running'))
   for j in st.get('running',[]):
    log=c/'runs'/j['job']/'train_log.csv';meta=log.parent/'run_meta.json';rows=list(csv.DictReader(log.open())) if log.exists() else [];quiet &= len(rows)<43 and meta.exists() and time.time()-meta.stat().st_mtime>10
  if quiet:break
  time.sleep(2)
 subprocess.run([GIT,'-C',str(R),'add','--',str(S.relative_to(R))],check=True)
 if subprocess.run([GIT,'-C',str(R),'diff','--cached','--quiet']).returncode:subprocess.run([GIT,'-C',str(R),'commit','-m',message],check=True)
def prepare():
 assert os.getuid()==1000 and S.stat().st_uid==1000
 if (S/'config.json').exists():return verify()
 d=read(R/'experiments/ac_dataset_v1/dataset_config.json');indices={k:d['indices'][k] for k in ['train','val']};units=[]
 for method,study,jobbase in [('B0','ac_qualification_20261024_window01','qualification_convnext_tiny_A'),('moments','ac_module_structure_window01','module_c4_moments'),('mom_corr','ac_crossscale_structure_window01','crossscale_mom_corr')]:
  for seed in [45,46,47]:
   root=R/'experiments'/study/'runs'/f'{jobbase}_s{seed}';c=read(root/'complete.json');assert c['epochs_completed']==60 and sha(R/c['checkpoint'])==c['checkpoint_sha256'];prediction=root/'validation_predictions.csv'
   units.append(dict(method=method,seed=seed,complete=str((root/'complete.json').relative_to(R)),complete_sha256=sha(root/'complete.json'),checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256'],signature=c['signature'],predictions=str(prediction.relative_to(R)),prediction_sha256=sha(prediction)))
 sources={p.name:sha(p) for p in S.iterdir() if p.suffix=='.py' or p.name=='TEXTURE_EVIDENCE_PROTOCOL_ZH.md'}
 cfg=dict(created_at=stamp(),identity=S.name,indices=indices,sources=sources,manifest_sha256=sha(R/'experiments/ac_dataset_v1/dataset_manifest.json'),data_config_sha256=sha(R/'experiments/ac_dataset_v1/dataset_config.json'),cache_meta_sha256=sha(R/'.cache_data_study/meta.json'),units=units,features=['mean','std','relative_fine_energy','gradient_anisotropy'],train_quantiles=[1/3,2/3],quantile_method='linear',seeds=[45,46,47],bootstrap=dict(replicates=20000,seed=20261025),neural_training=0,test_image_reads=0,source_model_selection_test_dependency=True,independent_generalization=False)
 save(S/'config.json',cfg);commit('research: preregister train-thresholded texture and validation-error evidence on nine frozen checkpoints');return verify()
def verify():
 cfg=read(S/'config.json');assert cfg['sources']=={p.name:sha(p) for p in S.iterdir() if p.suffix=='.py' or p.name=='TEXTURE_EVIDENCE_PROTOCOL_ZH.md'}
 assert sha(R/'experiments/ac_dataset_v1/dataset_manifest.json')==cfg['manifest_sha256'] and sha(R/'experiments/ac_dataset_v1/dataset_config.json')==cfg['data_config_sha256'] and sha(R/'.cache_data_study/meta.json')==cfg['cache_meta_sha256']
 for u in cfg['units']:assert sha(R/u['complete'])==u['complete_sha256'] and sha(R/u['predictions'])==u['prediction_sha256']
 return cfg
def features(gray):
 import numpy as np
 g=np.asarray(gray,dtype=np.float64);assert g.ndim==2
 padded=np.pad(g,2,mode='reflect');integral=np.pad(padded,((1,0),(1,0))).cumsum(0).cumsum(1);box=(integral[5:,5:]-integral[:-5,5:]-integral[5:,:-5]+integral[:-5,:-5])/25
 dx=(g[1:-1,2:]-g[1:-1,:-2])/2;dy=(g[2:,1:-1]-g[:-2,1:-1])/2;xx=np.mean(dx*dx);yy=np.mean(dy*dy);xy=np.mean(dx*dy);std=float(g.std())
 return dict(mean=float(g.mean()),std=std,relative_fine_energy=float(np.sqrt(np.mean((g-box)**2))/(std+1e-4)),gradient_anisotropy=float(np.sqrt((xx-yy)**2+4*xy**2)/(xx+yy+1e-8)))
def metrics(y,p,groups,weights=None):
 import numpy as np
 pred=p.argmax(1);w=np.ones(len(y)) if weights is None else weights;cm=np.zeros((3,3));np.add.at(cm,(y,pred),w);den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(3),where=den>0)
 patient=[float(np.average((pred==y)[groups==g],weights=w[groups==g])) for g in sorted(set(groups)) if w[groups==g].sum()>0]
 return dict(macro_f1=float(f1.mean()),accuracy=float(np.average(pred==y,weights=w)),malignant_recall=float(cm[1,1]/cm[1].sum()) if cm[1].sum() else None,patient_equal_accuracy=float(np.mean(patient)),confusion_matrix=cm.astype(int).tolist())
