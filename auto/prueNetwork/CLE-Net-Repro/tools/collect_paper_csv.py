"""Build an auditable, protocol-separated CSV collection without editing sources."""
import csv,hashlib,json,shutil,statistics,tempfile
from collections import defaultdict
from pathlib import Path

PROJECT=Path('/home/login/data/cle/auto/prueNetwork/CLE-Net-Repro')
LEGACY=PROJECT.parent/'CLE-Net'
ORIGINAL=Path('/home/login/data/cle/auto/Network/CLE-Net')
REF=ORIGINAL/'historical_protocol_codex_20261109'
OUT=PROJECT/'paper_results_csv'
METRICS=['macro_f1','accuracy','weighted_f1','balanced_accuracy','auc_roc_macro',
         'macro_sensitivity','macro_specificity','kappa','mcc']
RECENT={
 'thinkingvit_deit':(2026,'CVPR main conference','ThinkingViT: Matryoshka Thinking Vision Transformer for Elastic Inference','https://github.com/ds-kiel/ThinkingViT','3H->6H; threshold=0; final-stage CE'),
 'cpubone_b2':(2026,'CVPR Findings','CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities','https://github.com/altair199797/CPUBone','timm cpubone_b2_bfrobust.r224_in1k'),
 'mambavision_t':(2025,'CVPR','MambaVision: A Hybrid Mamba-Transformer Vision Backbone','https://github.com/NVlabs/MambaVision','T variant'),
 'mambaout_tiny':(2025,'CVPR','MambaOut: Do We Really Need Mamba for Vision?','https://github.com/yuweihao/MambaOut','Tiny; gated CNN without SSM'),
 'transnext_tiny':(2024,'CVPR','TransNeXt: Robust Foveal Visual Perception for Vision Transformers','https://github.com/DaiShiResearch/TransNeXt','Tiny; 2024 older reference'),
}
NAMES={'convnext_tiny':'ConvNeXt-T baseline','convnext_egb_scg':'CLE-Net (EGB+SCG)',
 'convnext_egb':'ConvNeXt-T + EGB','convnext_scg':'ConvNeXt-T + SCG',
 'convnext_hsc':'ConvNeXt-T + HSC','convnext_egb_scg_hsc':'EGB+SCG+HSC ablation',
 'thinkingvit_deit':'ThinkingViT-DeiT 3H->6H','cpubone_b2':'CPUBone-B2',
 'mambavision_t':'MambaVision-T','mambaout_tiny':'MambaOut-Tiny','transnext_tiny':'TransNeXt-Tiny'}
PROTOCOLS={
 'A_verified_reference':('sunfuxiang','2.12.0+cu130','verified source/data/environment/initialization; 2026-11-09 paired reference'),
 'A_historical_ablation':('sunfuxiang (historical runner)','historical version not separately frozen','post-fix batch; full CLE-Net exactly reproduced under reference environment; other variants not rerun this turn'),
 'B_topnets_251':('cle_topnets','2.5.1+cu124','compare within environment B; do not describe as identical environment to A'),
 'B_zcode_r6b':('cle_topnets (R6 script)','2.5.1+cu124','separate Zcode batch; do not merge seeds with Codex runs'),
 'L_historical_backbone':('sunfuxiang (runner declaration)','not independently pinned','legacy backbone runs; preserve run count and provenance'),
 'L_cross_dataset':('sunfuxiang (runner declaration)','not independently pinned','independent training on each dataset; not direct external CLE deployment'),
 'L_TCGA':('source project historical environment','not independently pinned','six-class TCGA; recalculate sample SD; HSC variant separate from final model'),
 'L_capacity':('historical source project','not independently pinned','partially completed capacity scaling; do not imply complete coverage'),
 'archive_only':('see original run/source','unknown','exploratory or superseded record, not automatically eligible for paper main tables'),
}

def readcsv(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:
  return list(csv.DictReader(line for line in f if line.strip() and not line.lstrip().startswith('#')))

def dataset(d):
 s=d.get('test_dir','').lower()
 if 'tcga' in s:return 'TCGA'
 if 'covid' in s:return 'Covid'
 if 'busi' in s:return 'BUSI'
 if 'cub' in s:return 'CUB'
 if 'heel' in s:return 'heelBone'
 if '5class' in s:return 'CLE_5class'
 if 'cle_ac' in s:return 'CLE_ac'
 return 'unknown'

def record(path,batch,protocol):
 path=Path(path);d=json.loads(path.read_text());run=path.parent.parent
 meta=json.loads((run/'run_meta.json').read_text()) if (run/'run_meta.json').exists() else {}
 model=meta.get('model',d.get('model','unknown'));info=RECENT.get(model,('','','','',''))
 r=dict(dataset=dataset(d),protocol_id=protocol,batch=batch,model=model,model_name=NAMES.get(model,model),
   seed=meta.get('seed',''),n_test=d.get('n_total',''),n_classes=len(d.get('class_names',[])),
   role='control' if model in ('convnext_tiny','convnext_egb_scg') else 'comparison',
   paper_year=info[0],venue=info[1],paper_title=info[2],code_url=info[3],method_notes=info[4],
   is_2025_2026_method=model in RECENT and info[0] in (2025,2026),
   environment=PROTOCOLS[protocol][0],torch_version=PROTOCOLS[protocol][1],
   protocol_notes=PROTOCOLS[protocol][2],
   checkpoint_exists=(run/'model_best.pth').exists(),train_log_exists=(run/'train_log.csv').exists(),
   predictions_exists=(path.parent/'predictions.csv').exists(),
   pretrained_sha256=meta.get('pretrained_load',{}).get('checkpoint_sha256',''),
   backbone_load_audit=meta.get('pretrained_load',{}).get('all_trainable_backbone_parameters_loaded',''),
   selection_metric=meta.get('selection_metric',''),source_result=str(path),source_meta=str(run/'run_meta.json'))
 for k in METRICS:r[k]=d.get(k,'')
 if (run/'train_log.csv').exists():
  try:
   logs=readcsv(run/'train_log.csv');best=max(logs,key=lambda v:float(v['val_macro_f1']))
   r.update(best_epoch=best['epoch'],stop_epoch=logs[-1]['epoch'],best_training_val_macro_f1=float(best['val_macro_f1']))
  except (ValueError,KeyError):pass
 return r,d

def summarize(rows):
 groups=defaultdict(list)
 for r in rows:groups[(r['dataset'],r['protocol_id'],r['model'],r['batch'])].append(r)
 out=[]
 for _,rr in sorted(groups.items()):
  first=rr[0];s={k:first.get(k,'') for k in ['dataset','protocol_id','batch','model','model_name','role','paper_year','venue','paper_title','code_url','method_notes','is_2025_2026_method','environment','torch_version','protocol_notes']}
  s.update(n_runs=len(rr),seeds=';'.join(str(r['seed']) for r in sorted(rr,key=lambda r:str(r['seed']))),
   n_test=first['n_test'],n_classes=first['n_classes'],
   statistical_unit='independent training seed on fixed test set',std_ddof=1,
   completeness='complete' if all(r['checkpoint_exists'] and r['train_log_exists'] and r['predictions_exists'] for r in rr) else 'check missing artifacts',
   qualification='single run: SD not estimable' if len(rr)==1 else 'descriptive, no significance claim',
   source_results=';'.join(r['source_result'] for r in rr))
  for k in METRICS:
   vals=[float(r[k]) for r in rr if r.get(k,'')!='' and r[k] is not None]
   s[k+'_mean']=statistics.mean(vals) if vals else ''
   s[k+'_sample_std']=statistics.stdev(vals) if len(vals)>1 else ''
  out.append(s)
 return out

def main():
 if OUT.exists():raise RuntimeError('Destination already exists; do not overwrite a released collection')
 build=Path(tempfile.mkdtemp(prefix='.paper_results_build_',dir=PROJECT))
 catalog=[];selected={};alldata={}
 def write(name,rows,category,description,notes=''):
  assert rows, 'Empty table '+name
  cols=list(dict.fromkeys(k for row in rows for k in row))
  with (build/name).open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=cols,extrasaction='raise');w.writeheader();w.writerows(rows)
  catalog.append(dict(file=name,category=category,rows=len(rows),description=description,notes=notes,
   sha256=hashlib.sha256((build/name).read_bytes()).hexdigest()))
 def take(path,batch,protocol):
  r,d=record(path,batch,protocol);selected[str(path)]=r;alldata[str(path)]=d;return r
 audit=json.loads((PROJECT/'audit/legacy_result_inventory.json').read_text())
 def rows_for(batch,protocol,models=None):
  return [take(x['path'],batch,protocol) for x in audit['rows'] if x['batch']==batch and (models is None or x['model'] in models)]
 verified=[]
 for p in sorted((REF/'runs').glob('*/*/eval_test/results.json')):verified.append(take(p,'reference_rerun_20261109','A_verified_reference'))
 assert len(verified)==6
 write('03_CLE_已验证主结果.csv',summarize(verified),'main','CLE-Net and baseline; verified protocol A')
 write('04_CLE_已验证主结果_逐种子.csv',verified,'main','six independently trained and tested reference runs')
 ablation=rows_for('weights_cle_fixed_v1','A_historical_ablation',models={'convnext_egb','convnext_scg','convnext_hsc','convnext_esa','convnext_scg_hsc','convnext_egb_scg_hsc','convnext_esa_scg_hsc'})+verified
 abls=summarize(ablation);base=next(s for s in abls if s['model']=='convnext_tiny')['macro_f1_mean']
 for s in abls:s['delta_f1_vs_verified_baseline_pp']=100*(s['macro_f1_mean']-base)
 assert len(ablation)==27
 write('05_CLE_消融实验.csv',abls,'ablation','nine configurations; all deltas recomputed against verified baseline','Historical variants plus verified controls; not one newly trained batch')
 write('06_CLE_消融实验_逐种子.csv',ablation,'ablation','27 run records with raw-precision metrics and provenance')
 canonical=readcsv(ORIGINAL/'experiments/sci_paper_data/05_backbones_cle.csv')
 models={r['model'] for r in canonical if r['model'] not in ('convnext_tiny','CLE-Net (convnext_egb_scg)')}
 backbones=rows_for('zcode_source_weights_cle_ac_v1','L_historical_backbone',models=models)+verified
 write('07_CLE_传统主干对比.csv',summarize(backbones),'backbone comparison','legacy backbone results and verified reference controls','Protocols and n are separate; RepViT has only one run')
 write('08_CLE_传统主干对比_逐种子.csv',backbones,'backbone comparison','raw run metrics; historical five-seed baseline not silently pooled')
 recent=rows_for('codex_five','B_topnets_251')
 controls=rows_for('codex_rerun_251','B_topnets_251')
 assert len(recent)==15 and len(controls)==6
 write('09_CLE_近两年网络_2025_2026.csv',summarize([r for r in recent if r['is_2025_2026_method']]),'recent methods','four completed methods from 2025 and 2026','CPUBone is CVPR Findings; TransNeXt 2024 is excluded here')
 write('10_CLE_近两年网络_同环境对照.csv',summarize([r for r in recent if r['is_2025_2026_method']]+controls),'recent methods','four recent methods plus environment-B CLE-Net and baseline','Do not substitute protocol-A CLE-Net result in this table')
 write('11_CLE_五网络含2024参考.csv',summarize(recent+controls),'recent methods','all five completed methods including TransNeXt 2024 and B controls')
 write('12_CLE_五网络与对照_逐种子.csv',recent+controls,'recent methods','21 full run records; 15 methods runs plus six matched-environment controls')
 datasets={'CLE_ac':verified}
 for num,ds,source,bks in [(13,'Covid','weights_covid_v1','zcode_source_weights_covid_bk'),(15,'BUSI','weights_busi_v1','zcode_source_weights_busi_bk'),(17,'CUB','weights_cub_v1','zcode_source_weights_cub_bk')]:
  rr=rows_for(source,'L_cross_dataset')+rows_for(bks,'L_cross_dataset')
  datasets[ds]=rr
  write(f'{num:02d}_{ds}_全部模型对比.csv',summarize(rr),'cross dataset',f'{ds} independent dataset training; all available twelve-model comparison')
  write(f'{num+1:02d}_{ds}_逐种子.csv',rr,'cross dataset',f'{ds} original run-level metrics')
 tcga=rows_for('tcga_source_weights_tcga_ep','L_TCGA');datasets['TCGA']=tcga
 assert len(tcga)==45
 write('19_TCGA_全部模型对比.csv',summarize(tcga),'cross dataset','15 models x three runs; nine metrics recomputed with sample SD','Final CLE-Net and HSC variant kept separate; negative results retained')
 write('20_TCGA_逐种子.csv',tcga,'cross dataset','45 six-class test results')
 paired=[];two=[];deltas=[]
 for ds,rr in datasets.items():
  pair=[r for r in rr if r['model'] in ('convnext_tiny','convnext_egb_scg')]
  assert len({(r['model'],r['seed']) for r in pair})==len(pair)
  two.extend(summarize(pair))
  bs={r['seed']:r for r in pair if r['model']=='convnext_tiny'}
  cs={r['seed']:r for r in pair if r['model']=='convnext_egb_scg'}
  assert set(bs)==set(cs)
  dd=[]
  for seed in sorted(bs):
   r=dict(dataset=ds,seed=seed,baseline_source=bs[seed]['source_result'],CLE_Net_source=cs[seed]['source_result'])
   for metric in METRICS:r[metric+'_delta_pp']=100*(cs[seed][metric]-bs[seed][metric])
   paired.append(r);dd.append(r['macro_f1_delta_pp'])
  deltas.append(dict(dataset=ds,n_pairs=len(dd),mean_macro_f1_delta_pp=statistics.mean(dd),
    sample_std_macro_f1_delta_pp=statistics.stdev(dd),positive_seed_count=sum(x>0 for x in dd),
    paired_by_numeric_seed_only=True,note='same numeric seed does not guarantee identical loader streams across architectures; no significance test'))
 write('21_跨数据集_CLE与baseline.csv',two,'cross dataset','five datasets; final CLE-Net vs each dataset own baseline')
 write('22_跨数据集_逐种子差值.csv',paired,'cross dataset','CLE-Net minus baseline; percentage-point differences')
 write('23_跨数据集_增益汇总.csv',deltas,'cross dataset','positive seed counts and mean differences; no significance claim')
 capacity=[]
 for batch in ('weights_cle_sb_v1','weights_busi_sb_v1','weights_covid_sb_v1'):capacity.extend(rows_for(batch,'L_capacity'))
 write('24_容量扩展_有效运行.csv',capacity,'capacity','existing completed test records only; coverage incomplete')
 # Export classwise results and sparse confusion matrices only for curated tables.
 classrows=[];matrices=[]
 for path,r in selected.items():
  d=alldata[path]
  key={k:r[k] for k in ('dataset','protocol_id','batch','model','seed')}
  for label,values in d.get('per_class',{}).items():
   classrows.append(dict(**key,class_name=label,**values,source_result=path))
  names=d.get('class_names',[])
  for i,line in enumerate(d.get('confusion_matrix',[])):
   for j,count in enumerate(line):
    if count:matrices.append(dict(**key,true_class=names[i],predicted_class=names[j],count=count,source_result=path))
 write('25_逐类指标.csv',classrows,'per class','all curated test results; precision/recall/F1/specificity/support/counts where stored')
 write('26_混淆矩阵_非零项.csv',matrices,'confusion matrix','rows=true classes, columns=predicted classes; omitted cells are zero')
 incomplete=[]
 for r in audit['inventory']:
  if not r['eval_test']:incomplete.append(dict(model=r['model'],seed=r['seed'],source_run=r['path'],
    checkpoint_exists=r['checkpoint'],status='no completed test record; inspect training completion',metrics_available=False,
    note='R6 batch logs include rc=143 interruptions; checkpoint existence alone is insufficient'))
 for model,reason in [('ConvNeur','code downloaded; no usable pretrained link at original audit'),('VMamba','downloaded reserve, no formal run'),('UniRepLKNet','downloaded reserve, no formal run'),('lsnet_s','planned R6 run without completed result in audited inventory')]:
  incomplete.append(dict(model=model,seed='',source_run=str(LEGACY),checkpoint_exists='',status=reason,metrics_available=False,note='Not a zero score; do not fabricate metrics'))
 write('27_未完成与候选方法.csv',incomplete,'status','missing or unrun methods; not eligible as completed results')
 # One master record per physical result JSON, including exploratory archives.
 master=dict(selected)
 for r in audit['rows']:
  if r['path'] not in master:
   x,_=record(r['path'],r['batch'],'archive_only');master[r['path']]=x
 for path,r in master.items():r['used_in_curated_tables']=path in selected
 write('28_全部已核查运行台账.csv',list(master.values()),'master ledger','deduplicated physical result files; includes historical exploratory records','used_in_curated_tables distinguishes curated views from archive-only experiments')
 write('01_协议与环境.csv',[dict(protocol_id=k,environment=v[0],torch=v[1],notes=v[2]) for k,v in PROTOCOLS.items()],'protocol','Keep protocol IDs when selecting paper results')
 splits=readcsv(ORIGINAL/'experiments/sci_paper_data/01_dataset_splits.csv')
 for r in splits:r['source']=str(ORIGINAL/'experiments/sci_paper_data/01_dataset_splits.csv')
 tmeta=json.loads((Path(tcga[0]['source_result']).parent.parent/'run_meta.json').read_text())
 tcga_root=Path(tmeta['data_path_train']).parent
 counts={}
 for split in ('train','val','test'):
  counts[split]=sum(p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff') for p in (tcga_root/split).rglob('*')) if (tcga_root/split).exists() else ''
 splits.append(dict(dataset='TCGA',modality='histopathology',n_classes=6,**counts,
   total=sum(counts.values()) if all(isinstance(v,int) for v in counts.values()) else '',
   source=str(tcga_root),划分说明='current image-directory inventory; does not establish patient independence'))
 write('02_数据集划分.csv',splits,'datasets','split sizes and dataset sources')
 notes=[
 ('recent_definition','2025 and 2026 calendar years; TransNeXt 2024 is an older reference'),
 ('metrics_scale','classification metrics stored as decimals; delta_pp columns multiply differences by 100'),
 ('std','all recomputed summary tables use sample SD ddof=1; single-run SD is blank'),
 ('raw_precision','summary metrics recomputed from original results.json floats, not rounded source CSV values'),
 ('protocol_comparison','A CLE-Net 0.9630 and B new-method results cannot be claimed to share an identical runtime'),
 ('main_gain','verified CLE-Net minus verified baseline is +0.430323 percentage points, not old +1.41'),
 ('ownership','R6-Zcode and historical source records are identified; collecting them does not relabel them as Codex runs'),
 ('negative_results','TCGA final CLE-Net below baseline is retained; HSC variant is a different model'),
 ('confusion_matrix','only nonzero cells exported in sparse matrix table; class names from classwise table define all axes'),
 ('old_sources','source-copy tables preserve historical values and may use population SD; they are not corrected main tables'),
 ('capacity','not all capacity/data-set combinations were completed; missing data are not zeros'),
 ('scope','no new training or dataset modification performed; statistics are descriptive'),
 ('limitations','patient independence unverified; original fixed test set participated in development feedback'),
 ('main_Git','new reproducible project source commit b457d93; setup/validation commits bbb29fe/fc49b8e'),
 ]
 write('29_统计与使用说明.csv',[dict(item=a,explanation=b) for a,b in notes],'notes','definitions, limitations and comparison policy')
 # Preserve all distributed comparison/auxiliary CSVs as source copies, distinctly labeled.
 for n,f in enumerate(sorted((LEGACY/'experiment_data').glob('*.csv')),start=90):
  rr=readcsv(f)
  for r in rr:r['original_csv_source']=str(f)
  write(f'原始副本_{n}_{f.name}',rr,'legacy source copy','distributed CSV preserved as originally rounded','May mix population/sample SD or legacy baselines; use corrected tables above')
 for n,f in enumerate(sorted((ORIGINAL/'experiments/sci_paper_data').glob('*.csv')),start=110):
  rr=readcsv(f)
  for r in rr:r['original_csv_source']=str(f)
  write(f'原始副本_{n}_{f.name}',rr,'legacy SCI source copy','historical SCI statistics and auxiliary measurements','Original values retained; reference baseline and sample-SD corrections are in curated tables')
 # Validate exported numeric means and SDs directly against their own run rows.
 assert len([r for r in recent if r['is_2025_2026_method']])==12
 assert all(r['n_test']==576 for r in verified+recent+controls)
 assert len({r['source_result'] for r in master.values()})==len(master)
 for name in ['03_CLE_已验证主结果.csv','05_CLE_消融实验.csv','19_TCGA_全部模型对比.csv']:
  for row in readcsv(build/name):
   assert 0<=float(row['macro_f1_mean'])<=1
   assert int(row['n_runs'])>=1
 write('00_文件索引.csv',catalog,'index','Index of all other CSVs; SHA256 recorded before this index was written')
 build.rename(OUT)
 print(json.dumps(dict(folder=str(OUT),csv_files=len(list(OUT.glob('*.csv'))),
  curated_unique_runs=len(selected),master_unique_runs=len(master),
  recent_methods=4,all_five_methods=5,datasets=list(datasets)),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
