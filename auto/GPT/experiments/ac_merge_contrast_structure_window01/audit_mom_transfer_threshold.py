"""Post-result independent CPU verification; original threshold was already frozen before training."""
from pathlib import Path
from fractions import Fraction
import csv,datetime,hashlib,json,os
import numpy as np
R=Path('/home/login/data/cle/auto/GPT').resolve();S=R/'experiments/ac_merge_contrast_structure_window01';T=R/'experiments/ac_test_feedback_window01'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def f1(cm):return sum((Fraction(2*int(cm[k,k]),int(cm[k].sum()+cm[:,k].sum())) if cm[k].sum()+cm[:,k].sum() else Fraction() for k in range(3)),Fraction())/3
def counts(pred,y):
 cm=np.zeros((3,3),dtype=np.int64);np.add.at(cm,(y,pred),1);return cm
def main():
 assert os.getuid()==1000 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 cfg=read(S/'config.json');assert sha(S/'config.json')=='1fdd3b28feff6181e6096897011ce96b58744a3e94b31fb37ab08bfb501c7355'
 for p,h in {**cfg['sources'],**cfg['locked']}.items():assert sha(R/p)==h,p
 ids=cfg['indices']['test'];rec={z['index']:z for z in read(R/'experiments/ac_dataset_v1/dataset_manifest.json')};y=np.array([rec[i]['label'] for i in ids]);g=np.array([rec[i]['group_id'] for i in ids]);unique=sorted(set(g));assert len(ids)==576 and len(unique)==41
 seen={};weights_manifest=[]
 def predictions(p):
  z=list(csv.DictReader(p.open()));assert [int(r['index']) for r in z]==ids and [int(r['label']) for r in z]==y.tolist() and [r['group_id'] for r in z]==g.tolist();prefix='' if 'p0' in z[0] else 'mean_';v=np.array([[float(r[f'{prefix}p{k}']) for k in range(3)] for r in z]);assert np.isfinite(v).all() and np.allclose(v.sum(1),1,atol=1e-6) and v.min()>=0 and v.max()<=1;seen[str(p.relative_to(R))]=sha(p);return v.argmax(1)
 def check_weight(unit,role):
  assert sha(R/unit['checkpoint'])==unit['checkpoint_sha256'];weights_manifest.append(dict(role=role,seed=unit['seed'],checkpoint=unit['checkpoint'],sha256=unit['checkpoint_sha256']))
 group=read(S/'evaluations/mom_only/group_audit.json');reg=read(S/'evaluations/mom_only/registration.json');assert group['passed'] and group['registration_sha256']==sha(S/'evaluations/mom_only/registration.json') and reg['config_sha256']==sha(S/'config.json')
 for u,a in zip(reg['units'],group['units']):
  check_weight(u,'new_mom_Swin');folder=S/'runs'/u['job'];comp=read(folder/'complete.json');logs=list(csv.DictReader((folder/'train_log.csv').open()));assert len(logs)==comp['epochs_completed']==60 and comp['best_epoch']==max(range(60),key=lambda k:float(logs[k]['val_macro_f1']))==u['best_epoch'];assert sha(folder/'complete.json')==u['complete_sha256'] and sha(folder/'train_log.csv')==u['train_log_sha256'];assert all(int(z['actual_train_count'])==int(z['unique_train_count'])==2077 and z['train_index_set_sha256']=='e8fe76bbcf6d8faf8dcb6f38a148341ffe81d10ac0cdc3e4dc38e9f8ccafd691' for z in logs)
  assert a['passed'] and a['checkpoint_sha256']==u['checkpoint_sha256'] and a['val_max_probability_error']<1e-6 and a['test_max_probability_error']<1e-6
  for n,h in a['prediction_files'].items():assert sha(S/'evaluations/mom_only'/f's{u["seed"]}'/n)==h
 refs={}
 for name,method in [('B0','qualification_convnext_tiny_A'),('Swin_B','qualification_swin_t_B'),('DINO','dino_baseline')]:
  info=read(T/'methods_v2'/f'{method}.json');assert [u['seed'] for u in info['units']]==[45,46,47]
  for u in info['units']:check_weight(u,name);assert sha(R/u['complete'])==u['complete_sha256']
  refs[name]=[predictions(T/'predictions'/u['fingerprint']/'test_predictions.csv') for u in info['units']]
  if name in ['B0','Swin_B','DINO']:
   root=R/'experiments/ac_orientation_reference_window01' if name=='Swin_B' else R/'experiments/ac_c4_budget_controls_window01/controls'/name
   for u in info['units']:assert read(root/'runs'/f's{u["seed"]}'/'complete.json')['checkpoint_sha256']==u['checkpoint_sha256']
   refs[name+'_C4']=[predictions(root/'runs'/f's{s}'/'test_predictions.csv') for s in [45,46,47]]
 draws=np.random.default_rng(20261024).multinomial(41,np.full(41,1/41),size=20000)
 def conditional(a,b):
  values=[]
  for ps in [a,b]:
   cm=np.array([[counts(p[g==z],y[g==z]) for z in unique] for p in ps]);sampled=np.einsum('dg,sgij->dsij',draws,cm);tp=np.diagonal(sampled,axis1=-2,axis2=-1);den=sampled.sum(-1)+sampled.sum(-2);v=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0).mean((-1,-2));values.append(v)
   for n in range(5):
    index=np.concatenate([np.repeat(np.flatnonzero(g==z),int(w)) for z,w in zip(unique,draws[n])]);oracle=sum((f1(counts(p[index],y[index])) for p in ps),Fraction())/3;assert abs(float(oracle)-v[n])<1e-12
  return np.percentile(values[0]-values[1],[2.5,97.5]).tolist()
 result={};summary=read(S/'round_results.json')
 for mode in ['single','c4']:
  ps=[predictions(S/'evaluations/mom_only'/f's{s}'/f'test_{mode}_predictions.csv') for s in [45,46,47]];cms=[counts(p,y) for p in ps];scores=[f1(cm) for cm in cms];assert all(abs(float(v)-expected)<1e-12 for v,expected in zip(scores,summary['summary']['mom_only_'+mode]['seed_macro_f1']))
  for label,qs in refs.items():
   qcm=[counts(q,y) for q in qs];d=[x-f1(cm) for x,cm in zip(scores,qcm)];mean=sum(d,Fraction())/3;ci=conditional(ps,qs);gate=bool(mean>Fraction(3,200) and all(v>0 for v in d) and ci[0]>0)
   key=mode+'_vs_'+label;result[key]=dict(candidate_scores=[float(v) for v in scores],candidate_mean=float(sum(scores,Fraction())/3),paired_seed_deltas=[float(v) for v in d],exact_mean_delta=str(mean),mean_delta=float(mean),conditional_ci95=ci,strict_above_3over200=mean>Fraction(3,200),allseed_positive=all(v>0 for v in d),conditional_numerical_gate=gate,candidate_cm=[cm.tolist() for cm in cms],reference_cm=[cm.tolist() for cm in qcm])
   expected_label=dict(B0='qualification_convnext_tiny_A',Swin_B='qualification_swin_t_B',Swin_B_C4='Swin_C4',DINO='dino_baseline').get(label,label);old=summary['comparisons']['mom_only_'+mode+'_vs_'+expected_label];assert abs(old['mean_delta']-float(mean))<1e-12 and np.allclose(ci,old['conditional_patient_bootstrap']['ci95'],atol=1e-12,rtol=0) and gate==old['numerical_gate']
 out=dict(passed=True,checked_at=datetime.datetime.now().isoformat(),post_result_verification_not_new_preregistration=True,original_threshold_preregistered_sha256=sha(S/'comparison_preregistration.json'),source_sha256=sha(__file__),source_config_sha256=sha(S/'config.json'),group_audit_sha256=sha(S/'evaluations/mom_only/group_audit.json'),comparisons=result,prediction_sha256=seen,actual_weight_sha_verified=weights_manifest,bootstrap_draws=20000,bootstrap_direct_repeated_image_oracle_first5=True,patients=41,models_in_complete_round=15,models_finished_in_this_group=3,new_training=0,new_neural_inference=0,selection_bias_corrected=False,independent_generalization_confirmation=False,overall_research_complete=False)
 save(S/'independent_mom_transfer_threshold_audit.json',out)
 lines=['# 首组三seed：方向矩迁移与条件数值门槛独立复核','','这是一项结果后的CPU补充复核；原训练前冻结的门槛和配对bootstrap未改。首组三seed全部60epoch，最早valbest、源码/实际权重及完整预测SHA、427val/576test两路线一致性、2077索引均通过。剩余四arm继续原注册。','', '|比较|三seed差pp|均值差pp|41患者条件95%区间pp|严格数值门槛|','|---|---|---:|---|---|']
 for key,v in result.items():lines.append('|'+key+'|'+','.join(f'{100*x:.6f}' for x in v['paired_seed_deltas'])+f'|{100*v["mean_delta"]:.6f}|[{100*v["conditional_ci95"][0]:.6f},{100*v["conditional_ci95"][1]:.6f}]|{v["conditional_numerical_gate"]}|')
 lines+=['', 'mom_only C4对固定B0的数值门槛与新结构增益不是同一结论：单视图对同Swin父模型均值略负、CI跨0；同C4 Swin/B0/DINO差值CI也跨0。C4单模型116.769656832G，固定B0单视图8.909531136G，约13.1倍计算；同Swin C4父模型约114.391486464G。成熟SwinC4此前已越过B0门槛并通知，不能据此把现值自动称新增结构达标或独立泛化。','', '全部结果属于自适应内部开发，历史test查看/AC构建用标签、工作Swin B按历史test选择，患者bootstrap只是条件区间、不校正选择偏差。整数Fraction严格>3/200复算、首5次bootstrap直接重复图像oracle通过；不保证提点、SCI新颖或录用，不因本组/本轮结束停止研究。']
 (S/'MOM_TRANSFER_THRESHOLD_AUDIT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(passed=True,comparisons=result['c4_vs_B0'],matched_parent_C4=result['c4_vs_Swin_B_C4'],single_parent=result['single_vs_Swin_B']),ensure_ascii=False))
if __name__=='__main__':main()
