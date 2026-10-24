from pathlib import Path
import csv,hashlib,json,os,sys
import numpy as np
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import texture_evidence as e
def main():
 cfg=e.verify();os.nice(10);manifest={r['index']:r for r in e.read(e.R/'experiments/ac_dataset_v1/dataset_manifest.json')};allowed={k:set(v) for k,v in cfg['indices'].items()};assert len(allowed['train'])==2077 and len(allowed['val'])==427 and not allowed['train']&allowed['val']
 train_groups={manifest[i]['group_id'] for i in allowed['train']};val_groups={manifest[i]['group_id'] for i in allowed['val']};assert len(train_groups)==182 and len(val_groups)==39 and not train_groups&val_groups
 array=np.load(e.R/'.cache_data_study/resized_uint8.npy',mmap_mode='r');assert array.shape==(3595,256,256,3) and array.dtype==np.uint8;records=[];access=[]
 for split in ['train','val']:
  for index in cfg['indices'][split]:
   assert index in allowed[split] and manifest[index]['split']==split;rgb=array[index,16:240,16:240];gray=rgb.mean(2)/255.;f=e.features(gray);assert all(np.isfinite(list(f.values()))) and 0<=f['gradient_anisotropy']<=1+1e-12
   records.append(dict(index=index,split=split,group_id=manifest[index]['group_id'],label=manifest[index]['label'],**f));access.append(index)
 with (S/'image_features.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 thresholds={k:np.quantile([r[k] for r in records if r['split']=='train'],[1/3,2/3],method='linear').tolist() for k in cfg['features']};e.save(S/'train_thresholds.json',dict(thresholds=thresholds,source_config_sha256=e.sha(S/'config.json'),train_indices_sha256=hashlib.sha256(json.dumps(sorted(cfg['indices']['train'])).encode()).hexdigest(),derived_from_train_only=True))
 val=[r for r in records if r['split']=='val'];y=np.array([r['label'] for r in val]);groups=np.array([r['group_id'] for r in val]);pred={}
 for unit in cfg['units']:
  rows=list(csv.DictReader((e.R/unit['predictions']).open()));assert [int(r['index']) for r in rows]==cfg['indices']['val'] and [int(r['label']) for r in rows]==y.tolist() and [r['group_id'] for r in rows]==groups.tolist();p=np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows]);assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6);pred[(unit['method'],unit['seed'])]=p
 slices={};subset_masks={'all':np.ones(len(y),dtype=bool)}
 for feature in cfg['features']:
  bins=np.searchsorted(thresholds[feature],np.array([r[feature] for r in val]),side='right')
  for b in range(3):subset_masks[feature+'_'+['low','mid','high'][b]]=bins==b
 for name,mask in subset_masks.items():
  methods={}
  for method in ['B0','moments','mom_corr']:
   units=[dict(seed=seed,**e.metrics(y[mask],pred[(method,seed)][mask],groups[mask])) for seed in cfg['seeds']];methods[method]=dict(seeds=units,mean_macro_f1=float(np.mean([u['macro_f1'] for u in units])),mean_accuracy=float(np.mean([u['accuracy'] for u in units])))
  paired={}
  for a,b in [('moments','B0'),('mom_corr','moments')]:
   units=[]
   for seed in cfg['seeds']:
    ca=pred[(a,seed)][mask].argmax(1)==y[mask];cb=pred[(b,seed)][mask].argmax(1)==y[mask];units.append(dict(seed=seed,rescued=int(np.sum(ca&~cb)),harmed=int(np.sum(~ca&cb)),macro_f1_delta=methods[a]['seeds'][cfg['seeds'].index(seed)]['macro_f1']-methods[b]['seeds'][cfg['seeds'].index(seed)]['macro_f1']))
   paired[a+'_vs_'+b]=dict(seeds=units,mean_rescued=float(np.mean([u['rescued'] for u in units])),mean_harmed=float(np.mean([u['harmed'] for u in units])))
  slices[name]=dict(images=int(mask.sum()),patients=len(set(groups[mask])),class_counts=np.bincount(y[mask],minlength=3).tolist(),methods=methods,paired=paired)
 unique=np.unique(groups);patient_cm={}
 for key,p in pred.items():
  cm=np.zeros((len(unique),3,3),dtype=int)
  for slot,g in enumerate(unique):np.add.at(cm[slot],(y[groups==g],p[groups==g].argmax(1)),1)
  patient_cm[key]=cm
 def f(cm):
  den=cm.sum(-1)+cm.sum(-2);return np.divide(2*np.diagonal(cm,axis1=-2,axis2=-1),den,out=np.zeros_like(den,dtype=float),where=den>0).mean(-1)
 rng=np.random.default_rng(cfg['bootstrap']['seed']);draws=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=20000);boot={}
 for a,b in [('moments','B0'),('mom_corr','moments')]:
  d=np.stack([f(np.einsum('dp,pij->dij',draws,patient_cm[(a,seed)]))-f(np.einsum('dp,pij->dij',draws,patient_cm[(b,seed)])) for seed in cfg['seeds']]).mean(0);boot[a+'_vs_'+b]=dict(ci95=np.quantile(d,[.025,.975]).tolist(),draws=20000,patients=39,paired_seeds=3,first5_draw_counts=draws[:5].tolist(),first5_delta_values=d[:5].tolist(),scope='conditional validation description; adaptive model selection remains')
 e.save(S/'result.json',dict(slices=slices,full_val_conditional_bootstrap=boot,thresholds=thresholds,source_config_sha256=e.sha(S/'config.json'),neural_training=0,test_images=0,test_probabilities=0,multiple_exploratory_slices=True,independent_confirmation=False))
 e.save(S/'actual_access.json',dict(indices=access,train=2077,val=427,test=0,total_unique=len(set(access)),physical_cache_rows=3595,source_config_sha256=e.sha(S/'config.json')))
 lines=['# CLE纹理与验证错误：探索性CPU证据','','原固定B0、M方向矩和新跨尺度相关各三seed，均复用原val最高权重；没有新网络训练。图像仅固定2077train与427val，test图像和逐图概率零访问。原模型选择已有历史test依赖，不称独立验证/OOF。','', '|图像统计组|图数/患者|B0平均F1|方向矩平均F1|跨尺度平均F1|方向矩纠正/损伤均值|跨尺度对方向矩纠正/损伤|','|---|---|---:|---:|---:|---|---|']
 for name,v in slices.items():
  a=v['paired']['moments_vs_B0'];b=v['paired']['mom_corr_vs_moments'];lines.append(f'|{name}|{v["images"]}/{v["patients"]}|'+ '|'.join(f'{100*v["methods"][k]["mean_macro_f1"]:.4f}%' for k in ['B0','moments','mom_corr'])+f'|{a["mean_rescued"]:.2f}/{a["mean_harmed"]:.2f}|{b["mean_rescued"]:.2f}/{b["mean_harmed"]:.2f}|')
 lines+=['', '阈值全部只由train无标签分位数计算；四特征×三箱全列，分箱不改变原等级/清单。统计含背景，不等于临床质量或组织病理。少量错误、多重探索、已用val选择权重，不能把某一分箱更好解释为因果或可靠创新。完整每seed指标/混淆矩阵/恶性召回/患者等权准确率及全39患者条件区间见result.json。当前正式结构轮全部冻结控制继续，下一步应结合全部累计/去一/成熟参照，不据本表随意枚举结构。']
 (S/'TEXTURE_ERROR_REPORT_ZH.md').write_text('\n'.join(lines)+'\n');e.commit('results: preserve all train-thresholded validation texture-error slices and access evidence')
if __name__=='__main__':main()
