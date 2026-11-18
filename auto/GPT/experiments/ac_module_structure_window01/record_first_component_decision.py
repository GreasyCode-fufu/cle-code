"""Retain the first complete paired mechanism decision; do not change frozen jobs."""
from pathlib import Path
import csv,json,sys,time
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
m.verify();result=m.READ(S/'round_results.json');group=m.READ(S/'evaluations/wtconv/group_audit.json');assert group['passed']
single=result['comparisons']['wtconv_single_vs_qualification_convnext_tiny_A'];matched=result['comparisons']['wtconv_c4_vs_B0_C4']
assert single['paired_seed_deltas'][2]<0 and single['conditional_patient_bootstrap']['ci95'][0]<0<single['conditional_patient_bootstrap']['ci95'][1]
decision=dict(created_at=m.core.timestamp(),component='faithful WTConv at registered stages2/3',status='not_retained_as_confirmed_effective_component',reason='Singleview three-seed gain is inconsistent and patient interval spans zero; matched C4 gain likewise uncertain. This does not reject all wavelet designs.',singleview_vs_frozen_B0=single,c4_vs_matched_B0_c4=matched,all_seed_macro_f1=result['summary']['wtconv_single']['seed_macro_f1'],source_group_audit_sha256=m.sha(S/'evaluations/wtconv/group_audit.json'),source_comparison_snapshot=result['updated_at'],existing_jobs_changed=False,baseline_changed=False,overall_research_complete=False)
m.SAVE(S/'component_decisions/wtconv_round01.json',decision)
text='''# 首个共享模块三seed结论：WTConv

注册位置、算子、预训练与训练食谱保持不变；三个seed均完整60epoch并通过427val、576test逐图重建和患者配对审计。

|推理|seed45 test宏F1|seed46|seed47|均值|
|---|---:|---:|---:|---:|
|单视图|95.9562%|96.1452%|94.0083%|95.3699%|
|固定C4|96.4943%|96.1438%|95.2540%|95.9640%|

单视图相对冻结B0平均+0.3884百分点，seed47下降1.4587百分点，41患者条件95%区间[-0.4923,+1.4418]百分点；未达到+1.5，也未确认跨seed稳定收益。

同C4预算下相对B0 C4平均仅+0.0376百分点，区间[-0.8284,+0.7035]百分点。不将本次WTConv作为已证明有效的组件继续堆叠，也不因此否定所有小波或多尺度设计。

全部强参照比较、accuracy、恶性召回、患者等权accuracy、逐seed混淆矩阵、预算及逐图预测保存在round_results.json/evaluations/wtconv。首个实际完成checkpoint的单图使用脚本已独立核验；计算正确性不等于机制有效性。

其余注册模块继续运行。待配对证据支持有效组件后，再保留该组件并设计互补机制与重训消融。普通阴性不会成为总体研究停止条件。test为用户要求的适应性开发反馈，条件区间不校正选择偏差，不声称独立泛化确认。
'''
(S/'FIRST_COMPONENT_RESULT_ZH.md').write_text(text,encoding='utf-8')
while True:
    st=m.READ(S/'status.json');safe=bool(st.get('running'))
    for item in st.get('running',[]):
        p=Path('/proc')/str(item['pid'])/'cmdline';log=S/'runs'/item['job']/'train_log.csv';meta=log.parent/'run_meta.json';rows=list(csv.DictReader(log.open())) if log.exists() else []
        safe &= p.exists() and item['job'].encode() in p.read_bytes() and len(rows)<50 and meta.exists() and time.time()-meta.stat().st_mtime>10
    if safe or (S/'independent_audit.json').exists():break
    assert not st.get('failures');time.sleep(5)
m.commit('analysis: retain all paired WTConv results and decline to label unstable gains as a proven component')
print(json.dumps(dict(singleview_mean_delta=single['mean_delta'],matched_c4_mean_delta=matched['mean_delta'],retained=False)))
