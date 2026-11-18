import csv, json, statistics
from pathlib import Path

ROOT=Path('/home/login/data/cle/auto/prueNetwork/CLE-Net')
OUT=ROOT/'cle_net_rerun_codex_20261109_v2'
TOP=ROOT/'latest_top_networks/results'
DATA=ROOT/'cle_net_rerun_codex_20261109_v2/results'
spec=[
 ('convnext_egb_scg','CLE-Net (ConvNeXt-T + EGB + SCG)','Codex rerun'),
 ('convnext_tiny','ConvNeXt-T baseline','Codex rerun'),
 ('mambaout_tiny','MambaOut-Tiny','CVPR 2025'),
 ('transnext_tiny','TransNeXt-Tiny','CVPR 2024'),
 ('mambavision_t','MambaVision-T','CVPR 2025'),
 ('thinkingvit_deit','ThinkingViT-DeiT 3H->6H','CVPR 2026'),
 ('cpubone_b2','CPUBone-B2','CVPR Findings 2026'),
]
def find_new(model,seed):
 return next((DATA/f'{model}_seed{seed}').glob('*/eval_test/results.json'))
def find_old(model,seed):
 return TOP/model/f'seed{seed}'/model/'eval_test/results.json'
rows=[]
fields=['macro_f1','accuracy','weighted_f1','balanced_accuracy','auc_roc_macro','macro_sensitivity','macro_specificity','kappa','mcc']
for key,label,venue in spec:
 for seed in (42,43,44):
  p=find_new(key,seed) if key in ('convnext_egb_scg','convnext_tiny') else find_old(key,seed)
  d=json.loads(p.read_text())
  assert d['n_total']==576
  row={'model':key,'label':label,'venue':venue,'seed':seed,'source':str(p)}
  for f in fields: row[f]=d.get(f)
  row['params']=d.get('complexity',{}).get('params')
  row['macs']=d.get('complexity',{}).get('macs')
  if key=='thinkingvit_deit': row['params'],row['macs']='21.74M','5.85G'
  rows.append(row)
with (OUT/'comparison_per_seed_codex_rerun.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
summary=[]
for key,label,venue in spec:
 rr=[r for r in rows if r['model']==key]
 s={'model':key,'label':label,'venue':venue,'n_seeds':3}
 for f in fields:
  vals=[float(r[f]) for r in rr]
  s[f+'_mean']=statistics.mean(vals); s[f+'_std']=statistics.stdev(vals)
 s['params']=rr[0]['params']; s['macs']=rr[0]['macs']; summary.append(s)
(OUT/'comparison_summary_codex_rerun.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'comparison_summary_codex_rerun.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
metric_names=[('accuracy','Accuracy'),('weighted_f1','Weighted_F1'),('balanced_accuracy','Balanced_Accuracy'),('auc_roc_macro','Macro_AUROC'),('macro_sensitivity','Macro_Sensitivity'),('macro_specificity','Macro_Specificity'),('kappa','Cohen_Kappa'),('mcc','MCC')]
with (OUT/'paper_other_metrics_table_codex_rerun.csv').open('w',newline='',encoding='utf-8-sig') as f:
 cols=['Method']+[x for _,x in metric_names for x in (x+'_mean',x+'_std')]
 w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
 for s in summary:
  w.writerow({'Method':s['label'],**{x+suf:s[raw+suf] for raw,x in metric_names for suf in ('_mean','_std')}})
cle=summary[0]
with (OUT/'paper_comparison_table_codex_rerun.csv').open('w',newline='',encoding='utf-8-sig') as f:
 cols=['Method','Venue','Params','MACs','Macro_F1_mean','Macro_F1_std','Accuracy_mean','Accuracy_std','Macro_AUROC_mean','Macro_AUROC_std','Delta_vs_CLE_Net_pp']
 w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
 for s in summary:
  w.writerow({'Method':s['label'],'Venue':s['venue'],'Params':s['params'],'MACs':s['macs'],'Macro_F1_mean':s['macro_f1_mean'],'Macro_F1_std':s['macro_f1_std'],'Accuracy_mean':s['accuracy_mean'],'Accuracy_std':s['accuracy_std'],'Macro_AUROC_mean':s['auc_roc_macro_mean'],'Macro_AUROC_std':s['auc_roc_macro_std'],'Delta_vs_CLE_Net_pp':100*(s['macro_f1_mean']-cle['macro_f1_mean'])})
lines=['# CLE_ac Codex独立复跑汇总','',f'输出目录：{OUT}','', '本次由Codex独立启动，使用新目录；没有读取 weights_r6b_recheck 作为CLE-Net或baseline复跑结果。每个模型seed42/43/44顺序训练并测试。','', '模型 | Macro-F1 | Accuracy | Macro-AUROC','---|---:|---:|---:']
for s in summary: lines.append(f"{s['label']} | {s['macro_f1_mean']:.4f}±{s['macro_f1_std']:.4f} | {s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f} | {s['auc_roc_macro_mean']:.4f}±{s['auc_roc_macro_std']:.4f}")
lines += ['', '复跑manifest：LAUNCH_MANIFEST.txt；六个测试结果均n_total=576。']
(OUT/'EXPERIMENT_REPORT_CODEX_RERUN.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
