"""Report the numerical milestone without substituting it for network innovation."""
from pathlib import Path
import csv,hashlib,io,json,sys,tarfile
import numpy as np
S=Path(__file__).resolve().parent;R=S.parents[1];sys.path.insert(0,str(S))
import run_orientation_reference as m
F=R/'experiments/ac_field_structure_window01';P=R/'experiments/ac_c4_moment_prototype_window01';E=R/'experiments/ac_patient_mechanism_evidence_window01'
cfg=m.verify();result=m.READ(S/'orientation_results.json');gate=m.READ(S/'numerical_gate_independent_audit.json');assert gate['passed'] and gate['result_sha256']==m.sha(S/'orientation_results.json')
prior=m.READ(F/'evidence/test_feedback_snapshot.json');ids=cfg['indices']['test'];records={r['index']:r for r in m.READ(S/'manifest.json')};y=np.array([records[i]['label'] for i in ids]);groups=np.array([records[i]['group_id'] for i in ids])
mean=[np.load(S/'runs'/f's{seed}'/'test_views.npy').astype(float).mean(0) for seed in cfg['seeds']];extended={}
for reference in ['dino_baseline','qualification_convnext_base_A','qualification_convnext_base_B']:
    entry=prior['summary'][reference];probs=[]
    for unit in entry['seeds']:
        path=R/'experiments/ac_test_feedback_window01/predictions'/unit['fingerprint']/'test_predictions.csv';audit=m.READ(path.parent/'independent_audit.json')
        assert audit['passed'] and m.sha(path)==audit['prediction_sha256'];rows=list(csv.DictReader(path.open()));assert [int(r['index']) for r in rows]==ids
        probs.append(np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows]))
    differences=np.array(result['summary']['test']['C4_seed_macro_f1'])-np.array(entry['seed_macro_f1'])
    extended[reference]=dict(seed_deltas=differences.tolist(),mean_delta=float(differences.mean()),conditional_patient_bootstrap=m.bootstrap(mean,probs,y,groups),reference_mean=entry['macro_f1'])
m.SAVE(S/'extended_reference_comparisons.json',dict(source_result_sha256=m.sha(S/'orientation_results.json'),comparisons=extended,primary_B0_unchanged=True,independent_confirmation=False))
base=prior['summary']['qualification_'+cfg['primary_baseline']];w0=prior['summary']['qualification_swin_t_B'];test=result['summary']['test'];val=result['summary']['val'];difference=result['comparisons']['test_C4_vs_frozen_B0']
proto=m.READ(P/'numeric_preflight_cpu384.json');e=m.READ(E/'patient_evidence_results.json');assert e['analysis_revision']==2
lines=['# 固定A–C条件数值里程碑与继续进行的结构目标','',
    '本研究继续进行。成熟四视角推理首次通过预先固定的test数值门槛；新的可学习网络结构、2–3个经证据支持的互补机制及SCI创新性仍未完成。不能将本页称为独立泛化确认、论文创新成立或整个任务完成。','',
    '## 固定主要比较','',
    'B0按此前已提交的公平资格验证规则冻结为ConvNeXt-Tiny224，三seed验证宏F1均值98.1031%；选择没有根据test降低或替换。工作主干W0为Swin-T384，其选择使用了test反馈，依赖已明确披露。','',
    '|seed|固定B0 test F1|C4成熟参照 test F1|配对差（百分点）|','|---|---:|---:|---:|']
for seed,b,c,d in zip(cfg['seeds'],base['seed_macro_f1'],test['C4_seed_macro_f1'],difference['paired_seed_deltas']):lines.append(f'|{seed}|{100*b:.4f}%|{100*c:.4f}%|{100*d:+.4f}|')
lo,hi=difference['conditional_patient_bootstrap']['ci95'];alo,ahi=gate['separate_audit_CI95']
lines +=[f'|平均|{100*base["macro_f1"]:.4f}%|{100*test["macro_f1"]:.4f}%|{100*difference["mean_delta"]:+.4f}|','',
    f'20,000次41患者配对bootstrap的95%条件区间为[{100*lo:+.4f}, {100*hi:+.4f}]百分点；独立的固定audit seed重新抽样区间[{100*alo:+.4f}, {100*ahi:+.4f}]。精确有理数F1复算确认平均差严格大于0.015，全部seed正。详见numerical_gate_independent_audit.json。','',
    f'C4平均accuracy={100*test["accuracy"]:.4f}%，恶性召回={100*test["malignant_recall"]:.4f}%，患者等权accuracy={100*test["patient_equal_accuracy"]:.4f}%。所有seed混淆矩阵、逐图概率、角度分歧、权重/源码SHA均保存。','',
    f'验证集同时如实报告：C4平均宏F1={100*val["macro_f1"]:.4f}%，原0度={100*np.mean(val["plain_seed_macro_f1"]):.4f}%，有所下降。没有改用test选epoch、改标签、改阈值或筛图；原val0度最早最佳checkpoint保持不变。','',
    '## 强参照和成本','', '|比较|平均差（百分点）|三个seed差值|','|---|---:|---|']
for name,c in extended.items():lines.append(f'|C4 vs {name}|{100*c["mean_delta"]:+.4f}|'+', '.join(f'{100*d:+.4f}' for d in c['seed_deltas'])+'|')
lines+=['','相对DINO的平均差未超过1.5百分点，且不是全部seed正，不能借本结果宣称全面稳定优于DINO。主要B0沿用冻结选择，不事后替换参照来改写结论。','',
    f'C4不增加参数，仍为{result["underlying_parameters"]:,}；每图进行4次主干forward，计算{result["counted_flops"]/1e9:.5f} GFLOPs，是原W0的4倍，不能把额外计算或成熟TTA包装为新网络收益。GFLOPs为2 FLOP/MAC，排除norm、激活、softmax、pool、逐元素和数据搬运。耗时来自与其它有效训练共享GPU的测量，非独占部署延迟基准。','',
    '## 局限与尚未完成的工作','',
    'test由用户明确授权参与方法开发；历史研究已查看test，A–C构建也用过其标签。条件bootstrap没有校正反复查看、方案选择或A–C过滤依赖。等级是模型一致性代理，不证明医生标签全部正确。固定集数值稳定达标不能转称全新外部验证或独立可靠泛化。','',
    '原几何支持完整H1三seed测试未获稳定收益，源/配置/权重/阴性保留，其忠实参照和消融仍在执行；患者等权原型CPU诊断也无稳定收益，不计作有效创新。该诊断bootstrap v1仅患者数元数据误写41，已保留原版，v2独立重建39患者抽样并确认原区间数值不变；没有重新提取或训练。','',
    '## 已实施的低成本结构原型','',
    '已独立实现C4方向矩stem：预训练stem与90/180/270度共享旋转滤波器的响应均值/标准差，经零初始化1×1投影加回原stem。全部原主干保留。它借用成熟group lifting/pooling，是否形成论文创新尚待相关工作和训练证据，不能仅因算子组合就作创新结论。','',
    f'CPU NumPy算术、C4辅助分支空间等变、随机置换负对照、flat区有限梯度、有限差分、RNG保留、原224/384预训练logits逐位相同均通过。参数增加{proto["new_parameters"]:,}至{proto["parameters"]:,}；384逻辑计算{proto["counted_flops"]/1e9:.5f} GFLOPs，相对一次原W0增加{100*proto["relative_flop_increase"]:.3f}%。这不是GPU实测延迟，也不是模型训练或性能收益。','',
    '后续正式训练仍须新目录冻结完整协议、同公开预训练/同60epoch/三配对seed、合理成熟旋转增强与算子消融、batch32 CUDA有限梯度/零初始化/原W0全427概率复算和实测资源。保留已验证的四方向成熟参照，分别比较新模块单视角及相同四视角下的结构增益。若新模块有效，再保留它增加1–2个互补机制并做累加/drop-one重训；普通阴性不结束整体研究。']
(S/'CONDITIONAL_MILESTONE_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');m.commit('report: disclose conditional C4 numerical milestone, stronger references, compute cost and unfinished network goals')
print('milestone_report_saved',S/'CONDITIONAL_MILESTONE_ZH.md',flush=True)
