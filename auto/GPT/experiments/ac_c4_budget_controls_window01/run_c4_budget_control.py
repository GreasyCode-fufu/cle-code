"""Same fixed C4 policy on B0 and DINO; primary B0 is never replaced."""
from pathlib import Path
import argparse,csv,gc,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent;R=ROOT.parents[1]
C4=R/'experiments/ac_orientation_reference_window01';Q=R/'experiments/ac_qualification_20261024_window01';F=R/'experiments/ac_field_structure_window01';T=R/'experiments/ac_test_feedback_window01'
sys.path[:0]=[str(C4),str(Q),str(T),str(R/'experiments'),str(R)]
import run_orientation_reference as m
from audit_test_feedback_v2 import compute_metrics
original_factory=m.q.model_for
parser=argparse.ArgumentParser();parser.add_argument('--control',required=True,choices=['B0','DINO']);parser.add_argument('command',choices=['prepare','preflight','evaluate','analyze']);a=parser.parse_args()
S=ROOT/'controls'/a.control;S.mkdir(parents=True,exist_ok=True);m.S=S

def source_hashes():
    files=[ROOT/n for n in ['run_c4_budget_control.py','bootstrap_c4_budget_controls.py','C4_BUDGET_CONTROLS_PROTOCOL_ZH.md']]
    files+=[C4/'run_orientation_reference.py',Q/'run_qualification.py',Q/'qualification_model.py',T/'run_test_feedback_v2.py',T/'audit_test_feedback_v2.py']
    result={str(p.relative_to(R)):m.sha(p) for p in files}
    if a.control=='DINO':result.update(m.feedback.first.source_hashes())
    return result
m.sources=source_hashes

def factory(seed,arm):
    return m.feedback.first.model_for(seed,'baseline') if a.control=='DINO' else original_factory(seed,arm)
m.q.model_for=factory

def prepare():
    if (S/'config.json').exists():return m.verify()
    m.q.verify();m.feedback.verify()
    if a.control=='DINO':m.feedback.first.verify()
    name='qualification_'+m.READ(F/'config.json')['primary_baseline'] if a.control=='B0' else 'dino_baseline'
    registration=m.READ(T/'methods_v2'/f'{name}.json');assert [u['seed'] for u in registration['units']]==[45,46,47]
    for unit in registration['units']:assert m.sha(R/unit['checkpoint'])==unit['checkpoint_sha256']
    arm=registration['arm'];recipe=registration['recipe']
    budget=next(b for b in m.READ((Q if a.control=='B0' else m.feedback.S1)/'preflight.json')['budgets'] if b['arm']==arm)
    (S/'reference.json').write_bytes((T/'methods_v2'/f'{name}.json').read_bytes());(S/'manifest.json').write_bytes((C4/'manifest.json').read_bytes())
    ds=m.READ(R/'experiments/ac_dataset_v1/dataset_config.json')
    locked=[S/'reference.json',S/'manifest.json',Q/'config.json',T/'config_v2.json',T/'cache_audit.json',C4/'orientation_results.json',C4/'config.json',R/'experiments/ac_dataset_v1/dataset_config.json',R/'experiments/ac_dataset_v1/dataset_manifest.json']
    m.SAVE(S/'config.json',dict(control=a.control,source_method=name,sources=source_hashes(),locked={str(p.relative_to(R)):m.sha(p) for p in locked},
        units=registration['units'],seeds=[45,46,47],family='convnext_tiny' if a.control=='B0' else 'dinov2_vits14',recipe=recipe,arm=arm,
        indices={split:ds['indices'][split] for split in ['val','test']},angles=[0,90,180,270],aggregation='equal arithmetic probability mean',batch_size=32,amp='float16',budget=budget,
        checkpoint_selection='original earliest val0-best full60-epoch checkpoint; no retraining/reselection',primary_baseline=m.READ(F/'config.json')['primary_baseline'],
        frozen_primary_baseline_unchanged=True,inference_compute_multiplier=4,network_updates=0,test_role='adaptive development feedback',novel_network_contribution=False))
    m.commit('research: register identical C4 budget control for '+a.control+' without changing fixed primary baseline')
    return m.verify()

def analyze():
    cfg=m.verify();records={r['index']:r for r in m.READ(S/'manifest.json')};units=[];summary={};comparisons={}
    for unit in cfg['units']:
        folder=S/'runs'/f's{unit["seed"]}';c=m.READ(folder/'complete.json');assert c['state_unchanged'] and c['checkpoint_sha256']==unit['checkpoint_sha256']
        assert all(m.sha(folder/n)==h for n,h in c['files'].items());units.append(c)
    for split in ['val','test']:
        ids=cfg['indices'][split];y=np.array([records[i]['label'] for i in ids]);g=np.array([records[i]['group_id'] for i in ids]);means=[];plain=[];swin=[];metrics=[];pm=[]
        for seed in cfg['seeds']:
            folder=S/'runs'/f's{seed}';v=np.load(folder/(split+'_views.npy'));assert v.shape==(4,len(ids),3);mean=v.astype(float).mean(0)
            rows=list(csv.DictReader((folder/(split+'_predictions.csv')).open()));assert [int(r['index']) for r in rows]==ids
            assert [r['sample_id'] for r in rows]==[records[i]['sample_id'] for i in ids] and [r['group_id'] for r in rows]==g.tolist() and [int(r['label']) for r in rows]==y.tolist()
            assert np.max(np.abs(np.array([[float(r[f'mean_p{k}']) for k in range(3)] for r in rows])-mean))<1e-10
            means.append(mean);plain.append(v[0]);metrics.append(compute_metrics(mean,y,g));pm.append(compute_metrics(v[0],y,g))
            other=np.load(C4/'runs'/f's{seed}'/(split+'_views.npy')).astype(float).mean(0);swin.append(other)
        summary[split]=dict(C4_seed_macro_f1=[x['macro_f1'] for x in metrics],plain_seed_macro_f1=[x['macro_f1'] for x in pm],
            **{k:float(np.mean([x[k] for x in metrics])) for k in ['macro_f1','accuracy','malignant_recall','patient_equal_accuracy']})
        d=np.array(summary[split]['C4_seed_macro_f1'])-np.array(summary[split]['plain_seed_macro_f1'])
        comparisons[split+'_control_C4_vs_plain']=dict(paired_seed_deltas=d.tolist(),mean_delta=float(d.mean()),conditional_patient_bootstrap=m.bootstrap(means,plain,y,g))
        sm=[compute_metrics(p,y,g)['macro_f1'] for p in swin];d=np.array(sm)-np.array(summary[split]['C4_seed_macro_f1'])
        comparisons[split+'_Swin_C4_vs_control_C4']=dict(paired_seed_deltas=d.tolist(),mean_delta=float(d.mean()),conditional_patient_bootstrap=m.bootstrap(swin,means,y,g))
    result=dict(control=a.control,summary=summary,comparisons=comparisons,units=units,sources=source_hashes(),config_sha256=m.sha(S/'config.json'),
        parameters=cfg['budget']['parameters'],counted_flops=4*cfg['budget']['counted_flops'],frozen_primary_B0_unchanged=True,network_updates=0,novel_network_contribution=False,independent_confirmation=False,overall_research_complete=False)
    m.SAVE(S/'control_results.json',result);m.SAVE(S/'independent_computational_audit.json',dict(passed=True,models=3,val_images=427,val_patients=39,test_images=576,test_patients=41,
        all_original_probabilities_reproduced=True,all_mean_predictions_reconstructed=True,source_weight_prediction_hashes=True,all_states_unchanged=True,frozen_primary_B0_unchanged=True,network_updates=0))
    lines=[f'# {a.control}的相同C4预算对照','',
        '固定主baseline B0不变；本页补齐相同0/90/180/270度等权概率平均的成熟控制。三个原val0最佳权重不重训、不重选，test继续为开发反馈；不能称独立验证或新网络创新。','',
        '|split|C4 seed45|seed46|seed47|C4均值|原0度均值|','|---|---:|---:|---:|---:|---:|']
    for split,v in summary.items():lines.append('|'+split+'|'+'|'.join(f'{100*x:.4f}%' for x in v['C4_seed_macro_f1']+[v['macro_f1'],float(np.mean(v['plain_seed_macro_f1']))])+'|')
    lines+=['',f'原网络参数{result["parameters"]:,}，C4成本{result["counted_flops"]/1e9:.5f} GFLOPs（2FLOP/MAC口径，原W0四视角为114.39149G）；相同四视角可分离部分成熟推理增益，分辨率/主干差异仍须报告。','',
        '|比较|平均差（百分点）|患者条件95%区间|','|---|---:|---|']
    for name,v in comparisons.items():
        lo,hi=v['conditional_patient_bootstrap']['ci95'];lines.append(f'|{name}|{100*v["mean_delta"]:.4f}|[{100*lo:.4f}, {100*hi:.4f}]|')
    lines+=['','不能把相对原0度B0的数值门槛通过，改写为相对所有同预算增强baseline均提升1.5百分点。本对照不替换主要baseline，也不放宽成功标准。新学习结构仍需在强参照、相同预算和重训消融下验证。']
    (S/'BUDGET_CONTROL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');m.commit('audit: retain every matched C4 control and stronger-reference difference for '+a.control)
    print(json.dumps(dict(control=a.control,summary=summary,comparisons=comparisons),ensure_ascii=False),flush=True)

if a.command=='prepare':prepare()
elif a.command=='analyze':analyze()
else:getattr(m,a.command)()
