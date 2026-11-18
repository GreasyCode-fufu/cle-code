"""Preregistered direct cumulative/drop-one control comparisons; no neural work."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,sys
import numpy as np
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
from audit_module_structure import a

def main():
    cfg=m.verify();assert m.READ(S/'independent_audit.json')['passed'];policy=m.READ(S/'comparison_preregistration.json');assert policy['source_config_sha256']==m.sha(S/'config.json') and policy['comparator_sha256']==m.sha(__file__)
    rows={r['index']:r for r in m.READ(m.R/'experiments/ac_dataset_v1/dataset_manifest.json')};ids=cfg['indices']['test'];y=np.array([rows[i]['label'] for i in ids]);groups=np.array([rows[i]['group_id'] for i in ids]);result={}
    def exact(pred):
        cm=np.zeros((3,3),dtype=int)
        for true,guess in zip(y,pred.argmax(1)):cm[true,guess]+=1
        f=sum((Fraction(2*int(cm[k,k]),int(cm[k].sum()+cm[:,k].sum())) for k in range(3)),Fraction())/3
        assert abs(float(f)-a.compute_metrics(pred,y,groups)['macro_f1'])<1e-12
        return f,cm.tolist()
    for mode in ['single','c4']:
        candidate_paths=[S/'evaluations/mom_focalcontrast'/f's{s}'/f'test_{mode}_predictions.csv' for s in m.SEEDS]
        def checked(path):
            records,pred=a.readpred(path);assert [int(r['index']) for r in records]==ids
            assert [int(r['label']) for r in records]==y.tolist() and [r['group_id'] for r in records]==groups.tolist()
            return pred
        candidate=[checked(p) for p in candidate_paths]
        for reference in ['M_moments','focalcontrast_only','mom_misaligned','mom_additive','mom_focalraw','B0']:
            if reference=='M_moments':paths=[m.R/'experiments/ac_module_structure_window01/evaluations/c4_moments'/f's{s}'/f'test_{mode}_predictions.csv' for s in m.SEEDS]
            elif reference=='B0':paths=[m.T/'predictions'/u['fingerprint']/'test_predictions.csv' for u in m.READ(S/'baseline_reference.json')['units']]
            else:paths=[S/'evaluations'/reference/f's{s}'/f'test_{mode}_predictions.csv' for s in m.SEEDS]
            control=[checked(p) for p in paths];d=[];matrices=[]
            for p,q in zip(candidate,control):
                fp,cp=exact(p);fq,cq=exact(q);d.append(fp-fq);matrices.append(dict(candidate=cp,reference=cq))
            mean=sum(d,Fraction())/3;ci=a.conditional_bootstrap(candidate,control,y,groups)
            result[mode+'_vs_'+reference]=dict(paired_seed_deltas=[float(v) for v in d],mean_delta=float(mean),exact_mean_fraction=str(mean),exact_strict_threshold=bool(mean>Fraction(3,200)),all_seeds_positive=all(v>0 for v in d),conditional_patient_bootstrap=ci,conditional_numerical_gate=bool(mean>Fraction(3,200) and all(v>0 for v in d) and ci['ci95'][0]>0),confusion_matrices=matrices,candidate_prediction_sha256=[m.sha(p) for p in candidate_paths],reference_prediction_sha256=[m.sha(p) for p in paths],independent_generalization_confirmation=False)
    m.SAVE(S/'direct_control_comparisons.json',dict(passed=True,comparisons=result,predeclared_comparisons=policy,test_role='adaptive internal development, no selection-bias adjustment',overall_research_complete=False))
    lines=['# 中间层聚焦对比直接控制比较','','每个新arm的完整三seed与原M方向矩/B0有效权重复用，不重训参考。独立整数Fraction复算宏F1；3/200仅历史数值字段，最终2–3网络设计累计+1.5，单点不设该准入门槛，区间只为适应性选择后的患者条件描述。','', '|比较|配对seed差（百分点）|平均差|41患者条件95%区间|','|---|---|---:|---|']
    for name,v in result.items():
        lo,hi=v['conditional_patient_bootstrap']['ci95'];lines.append(f'|{name}|'+','.join(f'{100*x:.4f}' for x in v['paired_seed_deltas'])+f'|{100*v["mean_delta"]:.4f}|[{100*lo:.4f},{100*hi:.4f}]|')
    (S/'DIRECT_CONTROL_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    m.commit('results: independently verify preregistered midstage focal contrast cumulative drop-one and spatial-statistic controls')
if __name__=='__main__':main()
