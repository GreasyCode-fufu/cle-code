import csv,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path('/home/login/data/cle/auto/prueNetwork/CLE-Net-Repro/five_networks_crossdatasets_20261110')
METRICS=['macro_f1','accuracy','weighted_f1','balanced_accuracy','auc_roc_macro','macro_sensitivity','macro_specificity','kappa','mcc']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 partial='--partial' in sys.argv
 status=json.loads((ROOT/'status.json').read_text())
 if not partial:assert status['phase']=='all_complete' and status['completed']==30,status
 completed={(r['dataset'],r['model'],int(r['seed'])) for r in csv.DictReader((ROOT/'per_seed.csv').open(encoding='utf-8-sig'))}
 protocol=json.loads((ROOT/'protocol.json').read_text());assert {d['name'] for d in protocol['datasets']}=={'BUSI','CUB'}
 for name,digest in json.loads((ROOT/'source_sha256.json').read_text()).items():assert sha(ROOT/'code'/name)==digest,name
 if not partial:
  for name,digest in json.loads((ROOT/'pretrained_sha256.json').read_text()).items():assert sha(ROOT/'code/latest_top_networks/pretrained'/name)==digest,name
  for image in json.loads((ROOT/'dataset_manifest.json').read_text()):assert sha(Path(image['path']))==image['sha256'],image['path']
 rows=[];audit=[];params={}
 for ds in protocol['datasets']:
  for model in protocol['models']:
   pp=json.loads((ROOT/f"preflight_{model}_{len(ds['classes'])}.json").read_text());params[ds['name']+'|'+model]=pp['params']
   for seed in protocol['seeds']:
    if partial and (ds['name'],model,seed) not in completed:continue
    base=ROOT/f"runs/{ds['name']}/{model}_seed{seed}";run=base/model;f=run/'eval_test/results.json';d=json.loads(f.read_text())
    assert (base/'TRAIN_DONE').exists() and (run/'model_best.pth').stat().st_size>0
    assert d['n_total']==ds['counts'][2] and d['class_names']==ds['classes']
    cm=d['confusion_matrix'];n=len(cm);total=sum(map(sum,cm));assert total==d['n_total'] and n==len(ds['classes'])
    f1=sum(2*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)) if sum(cm[i])+sum(row[i] for row in cm) else 0 for i in range(n))/n
    assert abs(f1-d['macro_f1'])<1e-12
    assert abs(sum(cm[i][i] for i in range(n))/total-d['accuracy'])<1e-12
    with (run/'eval_test/predictions.csv').open(encoding='utf-8-sig') as stream:assert sum(1 for _ in csv.DictReader(stream))==total
    meta=json.loads((run/'run_meta.json').read_text());assert meta['selection_metric']=='val_macro_f1'
    assert meta['pretrained_load']['all_trainable_backbone_parameters_loaded'] and not meta['pretrained_load']['unexpected_keys']
    rows.append(dict(dataset=ds['name'],model=model,seed=seed,**{m:d[m] for m in METRICS}))
    audit.append(dict(dataset=ds['name'],model=model,seed=seed,results=str(f),results_sha256=sha(f),checkpoint_sha256=sha(run/'model_best.pth'),predictions_sha256=sha(run/'eval_test/predictions.csv'),test_count=total,cm_f1_verified=True))
 summary=[]
 for ds in protocol['datasets']:
  for model in protocol['models']:
   rr=[r for r in rows if r['dataset']==ds['name'] and r['model']==model]
   if partial and not rr:continue
   if not partial:assert len(rr)==3
   s=dict(dataset=ds['name'],model=model,n=len(rr))
   for m in METRICS:s[m+'_mean']=statistics.mean(r[m] for r in rr);s[m+'_sd']=statistics.stdev(r[m] for r in rr) if len(rr)>1 else None
   summary.append(s)
 result=dict(per_seed=rows,summary=summary,audit=audit,params=params,protocol=protocol,excluded_dataset='Covid by explicit user instruction')
 (ROOT/('partial_results_audit.json' if partial else 'final_results_verified.json')).write_text(json.dumps(result,indent=2))
 print(f'VERIFIED {len(rows)} completed runs; source hashes, checkpoint presence, prediction counts and confusion matrices valid.')
if __name__=='__main__':main()
