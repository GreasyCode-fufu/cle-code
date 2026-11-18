"""Reconstruct all24 qualification results and select baseline from dev only."""
from pathlib import Path
import collections,csv,hashlib,json,sys
import numpy as np
import torch
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE)]
import run_qualification as q


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()


def state_sha(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()


def metrics(prob,y,groups):
 pred=prob.argmax(1)
 return dict(macro_f1=float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),accuracy=float(accuracy_score(y,pred)),
  malignant_recall=float(np.mean(pred[y==1]==1)),patient_equal_accuracy=float(np.mean([np.mean(pred[groups==g]==y[groups==g]) for g in sorted(set(groups))])),
  confusion_matrix=confusion_matrix(y,pred,labels=[0,1,2]).tolist())


def bootstrap(a,b,y,groups):
 unique=sorted(set(groups));rng=np.random.default_rng(20261024)
 weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=20000);values=[]
 for collection in [a,b]:
  cm=np.array([[confusion_matrix(y[groups==g],p.argmax(1)[groups==g],labels=[0,1,2]) for g in unique] for p in collection])
  sampled=np.einsum('dg,sgij->dsij',weights,cm)
  tp=np.diagonal(sampled,axis1=-2,axis2=-1);den=sampled.sum(-1)+sampled.sum(-2)
  values.append(np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0).mean((-1,-2)))
 d=values[0]-values[1]
 return dict(ci95=np.percentile(d,[2.5,97.5]).tolist(),draws=20000,groups=len(unique),
  note='Exploratory conditional dev comparison; selection/AC filtering and repeated val uncertainty remain.')


def main():
 cfg=q.verify();q.bind()
 cache=q.core.read_json(HERE/'cache448_meta.json');assert sha(HERE/'cache448/images.npy')==cache['sha256']
 asset=q.core.read_json(HERE/'pretrained_manifest.json')
 for f,info in asset['assets'].items():assert sha(ROOT/info['path'])==info['sha256']
 cohort=q.core.read_json(q.COHORT/'dataset_config.json');ac=q.core.read_json(q.COHORT/'dataset_manifest.json')
 assert cfg['indices']==cohort['indices']
 gs={s:{r['group_id'] for r in ac if r['split']==s} for s in ['train','val','test']}
 assert [len(gs[s]) for s in ['train','val','test']]==[182,39,41]
 assert all(not gs[a]&gs[b] for a,b in [('train','val'),('train','test'),('val','test')])
 records=q.core.read_json(HERE/'manifest.json');val=cfg['indices']['val'];y=np.array([records[i]['label'] for i in val]);groups=np.array([records[i]['group_id'] for i in val])
 work=[dict(job=j,folder=HERE/'runs'/j['id'],reused=False) for j in q.core.read_json(HERE/'jobs.json')]
 work += [dict(job=r,folder=HERE/'references'/f'convnext_base_A_s{r["seed"]}',reused=True) for r in cfg['references']]
 units=[];probs=collections.defaultdict(dict);budgets={x['arm']:x for x in q.core.read_json(HERE/'preflight.json')['budgets']}
 errors=collections.defaultdict(dict)
 for item in work:
  j=item['job'];folder=item['folder'];comp=q.core.read_json(folder/'complete.json')
  if item['reused']:
   assert comp['signature']==j['source_signature'] and all(sha(folder/n)==h for n,h in j['metadata_sha256'].items())
  else:
   assert comp['signature']==q.signature(j) and j['train']==cfg['indices']['train'] and j['select']==val
  assert sha(ROOT/comp['checkpoint'])==comp['checkpoint_sha256']
  log=list(csv.DictReader((folder/'train_log.csv').open()));assert len(log)==60 and [int(x['epoch']) for x in log]==list(range(60))
  best=max(range(60),key=lambda k:float(log[k]['val_macro_f1']));assert best==comp['best_epoch']
  indexsha=hashlib.sha256(json.dumps(sorted(cfg['indices']['train'])).encode()).hexdigest()
  assert all(int(row['actual_train_count'])==int(row['unique_train_count'])==2077 and row['train_index_set_sha256']==indexsha for row in log)
  saved=list(csv.DictReader((folder/'validation_predictions.csv').open()))
  assert [int(x['index']) for x in saved]==val and [int(x['label']) for x in saved]==y.tolist() and [x['group_id'] for x in saved]==groups.tolist()
  target=np.array([[float(x[f'p{k}']) for k in range(3)] for x in saved]);assert np.isfinite(target).all() and np.allclose(target.sum(1),1,atol=1e-6)
  q.ACTIVE_RECIPE=j['recipe'];model=q.model_for(j['seed'],j['arm']);state=torch.load(ROOT/comp['checkpoint'],map_location='cpu',weights_only=True)
  assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point());model.load_state_dict(state,strict=True)
  count=sum(p.numel() for p in model.parameters());assert count==comp['parameters']==budgets[j['arm']]['parameters']
  before=state_sha(model);p,actual_y,ix=q.core.infer(model,q.loader(val,False,j['seed']))
  assert ix==val and np.array_equal(actual_y,y) and state_sha(model)==before
  error=float(np.max(np.abs(p-target)));assert error<1e-6
  m=metrics(p,y,groups);assert abs(m['macro_f1']-comp['best_select_f1'])<1e-12 and abs(m['macro_f1']-float(log[best]['val_macro_f1']))<1e-12
  unit=dict(family=j['family'],recipe=j['recipe'],arm=j['arm'],seed=j['seed'],reused=item['reused'],metrics=m,parameters=count,
   counted_flops=budgets[j['arm']]['counted_flops'],checkpoint=comp['checkpoint'],checkpoint_sha256=comp['checkpoint_sha256'],
   prediction_sha256=sha(folder/'validation_predictions.csv'),best_epoch=best,seconds=comp['seconds'],max_probability_error=error)
  units.append(unit);probs[j['arm']][j['seed']]=p
  for pos,i in enumerate(val):errors[i][j['arm']+f'_s{j["seed"]}']=int(p[pos].argmax()!=y[pos])
  del state,model,p;torch.cuda.empty_cache()
 summary={}
 for family in q.FAMILIES:
  for recipe in q.RECIPES:
   arm=family+'_'+recipe;each=sorted([u for u in units if u['arm']==arm],key=lambda u:u['seed']);assert len(each)==3
   result=dict(family=family,recipe=recipe,arm=arm,seed_macro_f1=[u['metrics']['macro_f1'] for u in each],parameters=each[0]['parameters'],
    counted_flops=each[0]['counted_flops'],checkpoints=[{k:u[k] for k in ['seed','checkpoint','checkpoint_sha256']} for u in each])
   result.update({k:float(np.mean([u['metrics'][k] for u in each])) for k in ['macro_f1','accuracy','malignant_recall','patient_equal_accuracy']})
   result['qualified']=result['macro_f1']>=cfg['selection']['minimum_mean_macro_f1'] and min(result['seed_macro_f1'])>=cfg['selection']['minimum_seed_macro_f1']
   summary[arm]=result
 qualified=[x for x in summary.values() if x['qualified']]
 selected=None
 if qualified:
  highest=max(x['macro_f1'] for x in qualified)
  ties=[x for x in qualified if highest-x['macro_f1']<.0001]
  selected=min(ties,key=lambda x:(x['counted_flops'],x['parameters'],x['arm']))
 comparisons={}
 for family in q.FAMILIES:
  a=[probs[family+'_B'][s] for s in q.SEEDS];b=[probs[family+'_A'][s] for s in q.SEEDS]
  comparisons[family+'_B_vs_A']=dict(mean_delta=summary[family+'_B']['macro_f1']-summary[family+'_A']['macro_f1'],**bootstrap(a,b,y,groups))
 q.SAVE(HERE/'qualification_results.json',dict(summary=summary,units=units,recipe_comparisons=comparisons))
 frozen=dict(frozen=selected is not None,selected=selected,created_at=q.core.timestamp(),
  selection_rule=cfg['selection'],config_sha256=sha(HERE/'config.json'),source_hashes=q.sources(),
  warning='Chosen on repeatedly used dev/AC-selected cohort. No test inference or proven novel architecture. DINO retained as strong reference.')
 if (HERE/'selected_baseline.json').exists():
  previous=q.core.read_json(HERE/'selected_baseline.json')
  assert {k:v for k,v in previous.items() if k!='created_at'}=={k:v for k,v in frozen.items() if k!='created_at'},'Refuse changing already frozen baseline'
 else:q.SAVE(HERE/'selected_baseline.json',frozen)
 errorrows=[dict(index=i,label=records[i]['label'],group_id=records[i]['group_id'],**v) for i,v in sorted(errors.items())]
 q.core.save_csv(HERE/'validation_errors.csv',errorrows)
 wrong=[str(p) for p in [HERE,*HERE.rglob('*'),q.core.WEIGHTS,*q.core.WEIGHTS.rglob('*')] if p.stat().st_uid!=1000];assert not wrong
 q.SAVE(HERE/'independent_audit.json',dict(passed=True,timestamp=q.core.timestamp(),models=24,new_trainings=21,reused=3,test_images_loaded=0,
  selected_baseline_qualified=selected is not None,checks=['fixed AC and patient isolation','raw/cache/pretrained/source/checkpoint SHA',
   '60epochs/earliest maximum/actual training index sets','finite tensors/strict reload/full427 probability reproduction',
   'sklearn metrics/patient equal scores','independent patient-paired confusion bootstrap','Swin-inclusive compute budgets','all four family equal recipe/seed budget','login ownership']))
 lines=['# 其它成熟主干的公平资格比较结果','','本阶段为开发筛选，未运行新的test；不构成自主结构贡献。','',
  '| 流程 | 三seed宏F1 | 均值 | 恶性召回 | 患者等权准确率 | 合格 |','|---|---|---|---|---|---|']
 for arm,x in summary.items():
  lines.append('|'+arm+'|'+'/'.join(f'{v:.6f}' for v in x['seed_macro_f1'])+f'|{x["macro_f1"]:.6f}|{x["malignant_recall"]:.6f}|{x["patient_equal_accuracy"]:.6f}|{x["qualified"]}|')
 lines+=['','冻结基础baseline：'+(selected['arm'] if selected else '无合格者，不能挑弱baseline；须继续有依据的公平调优。'),
  '选择依据仅开发数据，DINO强参照及前一阶段阴性完整保留。下一步是该冻结基础上的CLE机制设计、重训消融与审计；整体目标仍在进行。',
  'AC构建及历史研究用过test标签/分数，反复开发与AC选择依赖必须披露，不能称未经接触外部验证。']
 (HERE/'QUALIFICATION_REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf8')
 print(json.dumps(dict(summary=summary,selected=selected),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
