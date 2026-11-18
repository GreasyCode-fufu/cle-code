"""Compose immutable full-probability auditor with explicit refinement comparisons."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
S=Path(__file__).resolve().parent;R=S.parents[1];sys.path.insert(0,str(S));import run_module_structure as m
old=R/'experiments/ac_module_structure_window01';spec=importlib.util.spec_from_file_location('frozen_full_probability_auditor',S/'independent_probability_auditor.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.S=S;a.m=m
base_summarize=a.summarize

def summarize():
    base_summarize();cfg=m.verify();summary=m.READ(S/'round_results.json');rows={r['index']:r for r in m.READ(R/'experiments/ac_dataset_v1/dataset_manifest.json')};ids=cfg['indices']['test'];y=np.array([rows[i]['label'] for i in ids]);g=np.array([rows[i]['group_id'] for i in ids]);comparisons={}
    reference_arms=['c4_moments','c4_aug','eca']
    for arm in m.ARMS:
        if not (S/'evaluations'/arm/'group_audit.json').exists():continue
        for mode in ['single','c4']:
            pred=[a.readpred(S/'evaluations'/arm/f's{s}'/f'test_{mode}_predictions.csv')[1] for s in m.SEEDS]
            for ref in reference_arms:
                rp=[a.readpred(old/'evaluations'/ref/f's{s}'/f'test_{mode}_predictions.csv')[1] for s in m.SEEDS]
                d=np.array([a.compute_metrics(p,y,g)['macro_f1']-a.compute_metrics(q,y,g)['macro_f1'] for p,q in zip(pred,rp)]);ci=a.conditional_bootstrap(pred,rp,y,g)
                comparisons[arm+'_'+mode+'_vs_M_'+ref]=dict(paired_seed_deltas=d.tolist(),mean_delta=float(d.mean()),conditional_patient_bootstrap=ci,independent_confirmation=False)
    m.SAVE(S/'mechanism_comparisons.json',dict(comparisons=comparisons,reference_round_config_sha256=m.sha(old/'config.json'),reference_round_audit_sha256=m.sha(old/'independent_audit.json'),original_B0_unchanged=True))
    lines=['# 局部细节强度对比：累计与消融对照','','配对参考复用原M轮同seed有效模型，不重复训练。所有原B0/强参照指标仍见MODULE_ROUND_REPORT_ZH.md。','', '|比较|平均差（百分点）|患者条件95%区间（百分点）|','|---|---:|---|']
    for k,v in comparisons.items():
        lo,hi=v['conditional_patient_bootstrap']['ci95'];lines.append(f'|{k}|{100*v["mean_delta"]:.4f}|[{100*lo:.4f}, {100*hi:.4f}]|')
    lines+=['','mom_detailcontrast对原Mom检验第一stage新增贡献；detailcontrast_only真去18528Mom；detailrawgate去q中心化；highpass仅D；misaligned固定错配q与D。最终2–3网络设计累计+1.5，单点以对父模型贡献评估。全部seed/阴性保留，不能据点估计称创新或独立确认。']
    (S/'MECHANISM_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
a.summarize=summarize
evaluate_completed_groups=a.evaluate_completed_groups

def final_audit():
    cfg=m.verify();assert len(m.READ(S/'jobs.json'))==15
    assert all((S/'evaluations'/arm/'group_audit.json').exists() for arm in m.ARMS)
    assert m.sha(R/'.cache_data_study/resized_uint8.npy')==m.READ(R/'.cache_data_study/meta.json')['cache_sha256']
    groups={split:{r['group_id'] for r in m.READ(R/'experiments/ac_dataset_v1/dataset_manifest.json') if r['split']==split} for split in ['train','val','test']}
    assert [len(groups[s]) for s in ['train','val','test']]==[182,39,41] and all(not groups[x]&groups[y] for x,y in [('train','val'),('train','test'),('val','test')])
    for arm in m.ARMS:
        folder=S/'evaluations'/arm;group=m.READ(folder/'group_audit.json');reg=m.READ(folder/'registration.json');assert group['passed'] and group['registration_sha256']==m.sha(folder/'registration.json')
        for unit in reg['units']:
            path=folder/f's{unit["seed"]}';audit=m.READ(path/'independent_audit.json');assert m.sha(R/unit['checkpoint'])==audit['checkpoint_sha256']
            assert all(m.sha(path/p)==h for p,h in audit['prediction_files'].items())
    assert all(p.stat().st_uid==1000 for root in [S,m.M,R/'weights/ac_local_detail_structure_window01'] for p in [root,*root.rglob('*')])
    m.SAVE(S/'independent_audit.json',dict(passed=True,models=15,seeds=m.SEEDS,train=2077,val=427,test=576,patient_isolation=True,earliest_val_best=True,actual_train_indices_exact=True,full_val_test_probability_reconstruction=True,source_and_weight_hashes=True,login_ownership=True,independent_generalization_confirmation=False,overall_research_complete=False));summarize()
