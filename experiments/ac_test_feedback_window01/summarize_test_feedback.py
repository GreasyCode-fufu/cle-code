"""Costs, resolution controls and patient errors from all completed paired feedback."""
from pathlib import Path
import csv,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_test_feedback_v2 as t
from audit_test_feedback_v2 import conditional_bootstrap

def main():
    report=t.read(HERE/'test_results.json');first={x['arm']:x for x in t.read(t.S1/'preflight.json')['budgets']}
    qual={x['arm']:x for x in t.read(t.Q/'preflight.json')['budgets']};costs={};probs={};errors=[]
    for name,m in report['summary'].items():
        method=t.read(HERE/'methods_v2'/f'{name}.json')
        b=(first if method['factory']=='first' else qual)[method['arm']]
        train_seconds=[];inference_seconds=[];audits=[];matrix=[];each=[]
        for u in method['units']:
            folder=HERE/'predictions'/u['fingerprint'];c=t.read(folder/'complete.json');a=t.read(folder/'independent_audit.json')
            assert a['passed'] and t.sha(folder/'test_predictions.csv')==a['prediction_sha256']
            rows=list(csv.DictReader((folder/'test_predictions.csv').open()));each.append(np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows]))
            comp=t.read(t.ROOT/u['complete']);train_seconds.append(comp['seconds']);inference_seconds.append(c['primary_inference_seconds']);audits.append(a['seconds']);matrix.append(a['metrics']['confusion_matrix'])
        probs[name]=each
        costs[name]=dict(parameters=b['parameters'],counted_flops=b['counted_flops'],gflops=b['counted_flops']/1e9,
            crop=224 if m['recipe']=='A' else 384,training_seconds_by_seed=train_seconds,
            primary_inference_seconds=inference_seconds,independent_reinference_seconds=audits,seed_confusion_matrices=matrix,
            note='Training elapsed is original run under its recorded concurrency, not an isolated speed benchmark. Reused DINO primary time excludes old inference; independent reinference is timed separately.')
    ac={r['index']:r for r in t.read(t.COHORT/'dataset_manifest.json') if r['split']=='test'};indices=t.read(HERE/'config_v2.json')['indices']
    y=np.array([ac[i]['label'] for i in indices]);groups=np.array([ac[i]['group_id'] for i in indices]);controls={}
    for family in t.qualification.FAMILIES:
        a='qualification_'+family+'_A';b='qualification_'+family+'_B'
        if a not in probs or b not in probs:continue
        delta=np.array(report['summary'][b]['seed_macro_f1'])-np.array(report['summary'][a]['seed_macro_f1'])
        controls[family]=dict(B384_minus_A224_seed_deltas=delta.tolist(),mean_delta=float(delta.mean()),
            patient_conditional_bootstrap=conditional_bootstrap(probs[b],probs[a],y,groups),
            compute_ratio=costs[b]['counted_flops']/costs[a]['counted_flops'],novel_network_component=False,
            independent_confirmation=False)
    for name,ps in probs.items():
        for group in sorted(set(groups)):
            positions=np.flatnonzero(groups==group);wrong=[float(np.mean(p[positions].argmax(1)!=y[positions])) for p in ps]
            errors.append(dict(method=name,group_id=group,images=len(positions),labels=','.join(map(str,sorted(set(y[positions])))),
                seed45_error_fraction=wrong[0],seed46_error_fraction=wrong[1],seed47_error_fraction=wrong[2],mean_error_fraction=float(np.mean(wrong))))
    t.save(HERE/'costs_and_resolution_controls.json',dict(created_at=t.core.timestamp(),test_results_sha256=t.sha(HERE/'test_results.json'),
        costs=costs,resolution_controls=controls,flop_convention='2 FLOP/MAC; conv/linear and explicit attention; excludes norm/activation/pooling/softmax/indexing/bias and memory',
        adaptive_test_feedback=True,selection_bias_corrected=False))
    t.core.save_csv(HERE/'patient_error_feedback.csv',errors)
    lines=['# test反馈的成本与下一轮依据','',
        '此处为用户授权的适应性开发分析。所有test反馈均留痕，不能作为独立泛化确认。患者错误只用于设计分析，不改标签、分级、阈值、清单或样本。','',
        '|方法|参数(M)|计算(GFLOPs)|crop|','|---|---:|---:|---:|']
    for name,c in costs.items():lines.append(f'|{name}|{c["parameters"]/1e6:.5f}|{c["gflops"]:.5f}|{c["crop"]}|')
    lines+=['','FLOPs按2 FLOP/MAC，包含显式注意力，其他排除项见JSON。不同原训练并发下的总耗时不能直接当独立速度基准比较。完整逐seed混淆矩阵、原始训练耗时、此次推理/独立复算耗时在成本JSON中。','',
        '|同主干384相对224|seed45差|seed46差|seed47差|平均差|患者条件95%区间|计算倍数|','|---|---:|---:|---:|---:|---|---:|']
    for family,c in controls.items():
        lo,hi=c['patient_conditional_bootstrap']['ci95'];vals='|'.join(f'{100*v:.4f}' for v in c['B384_minus_A224_seed_deltas'])
        lines.append(f'|{family}|{vals}|{100*c["mean_delta"]:.4f}|[{100*lo:.4f}, {100*hi:.4f}]|{c["compute_ratio"]:.3f}|')
    lines+=['','差值单位为百分点。分辨率变化是成熟训练设置，不冒称网络创新。条件区间不校正反复test选择。下一轮采用NEXT_MECHANISM_PREREGISTRATION_ZH.md中的固定B0/工作W0规则和几何视场支持H1，先正式CUDA数值与吞吐预检并提交再训练。',
        '','资格实验还未完成时，不锁定W0，不用暂时分数替换主要baseline；后续全部资格方法完成后需重新生成本附表。']
    (HERE/'COST_AND_NEXT_EVIDENCE_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    t.safe_commit('analysis: retain compute budgets, paired resolution feedback and all patient error fractions')
    print('costs and feedback evidence',len(costs),len(controls),len(errors),flush=True)

if __name__=='__main__':main()
