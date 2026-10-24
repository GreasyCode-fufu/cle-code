"""Current evidence bundle: conditional metric milestone, controls, negatives, new prototype."""
from pathlib import Path
import csv,hashlib,io,json,os,sys,tarfile
import numpy as np
S=Path(__file__).resolve().parent;R=S.parents[1]
NAMES=['ac_orientation_reference_window01','ac_c4_budget_controls_window01','ac_patient_mechanism_evidence_window01','ac_c4_moment_prototype_window01','ac_field_structure_window01']
paths={name:R/'experiments'/name for name in NAMES}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
C4=paths[NAMES[0]];BC=paths[NAMES[1]];E=paths[NAMES[2]];P=paths[NAMES[3]];F=paths[NAMES[4]]
main=read(C4/'orientation_results.json');gate=read(C4/'numerical_gate_independent_audit.json');assert gate['passed'] and sha(C4/'orientation_results.json')==gate['result_sha256']
b0=read(BC/'controls/B0/control_results.json');dino=read(BC/'controls/DINO/control_results.json');patient=read(E/'patient_evidence_results.json');assert patient['analysis_revision']==2
proto=read(P/'numeric_preflight_cpu_tiny224.json');assert proto['passed']
field=read(F/'round_results.json');status=read(F/'queue_status_v2.json');snapshot=read(F/'evidence/test_feedback_snapshot.json');base=snapshot['summary']['qualification_convnext_tiny_A']
lines=['# CLE新窗口研究：条件数值里程碑及结构研究进展','',
    '整体研究保持进行。成熟四方向推理通过相对预先冻结B0的固定test条件数值门槛；低成本可学习新网络和经消融支持的2–3个互补机制尚未完成。','',
    '## 三seed主要结果','', '|方法|seed45 test宏F1|seed46|seed47|均值|逻辑GFLOPs|','|---|---:|---:|---:|---:|---:|']
entries=[('固定B0 ConvNeXt-Tiny224 原0度',base['seed_macro_f1'],8.909531136),('同B0的成熟C4',b0['summary']['test']['C4_seed_macro_f1'],b0['counted_flops']/1e9),
    ('强参照DINO 原0度',snapshot['summary']['dino_baseline']['seed_macro_f1'],12.247128576),('DINO成熟C4',dino['summary']['test']['C4_seed_macro_f1'],dino['counted_flops']/1e9),
    ('工作参照Swin-T384 原0度',snapshot['summary']['qualification_swin_t_B']['seed_macro_f1'],28.597871616),('Swin-T384 成熟C4',main['summary']['test']['C4_seed_macro_f1'],main['counted_flops']/1e9)]
for name,values,cost in entries:lines.append('|'+name+'|'+'|'.join(f'{100*v:.4f}%' for v in values+[float(np.mean(values))])+f'|{cost:.5f}|')
c=main['comparisons']['test_C4_vs_frozen_B0'];lo,hi=c['conditional_patient_bootstrap']['ci95']
lines +=['',f'Swin成熟C4相对冻结B0平均差{100*c["mean_delta"]:+.4f}百分点，所有seed正；41患者20,000次配对95%条件区间[{100*lo:+.4f}, {100*hi:+.4f}]百分点。精确有理数F1和另一固定audit seed的独立重采样均复核通过。','',
    'B0早已按公平资格实验的val规则冻结（val均值98.1031%），不按test降低或替换。C4是成熟多视图推理，四倍原主干forward，不算新网络创新。','',
    '## 相同增强和强参照的预算对照','', '|比较|平均差（百分点）|患者条件95%区间|','|---|---:|---|']
for label,control in [('Swin C4 vs B0 C4',b0),('Swin C4 vs DINO C4',dino)]:
    v=control['comparisons']['test_Swin_C4_vs_control_C4'];lo,hi=v['conditional_patient_bootstrap']['ci95'];lines.append(f'|{label}|{100*v["mean_delta"]:+.4f}|[{100*lo:+.4f}, {100*hi:+.4f}]|')
lines +=['','以上同样增强的区间跨0，不能宣称Swin C4稳定优于这些增强强参照，更不能改写成相对它们都提升超过1.5百分点。分辨率和FLOPs仍不同，原始B0定义与数值结果保留。','',
    f'验证集也完整披露：Swin C4 val宏F1均值{100*main["summary"]["val"]["macro_f1"]:.4f}%，低于原0度；B0 C4与DINO C4的val结果也见各完整报告，未据test重选epoch。','',
    '## 新结构与阴性结果','',f'几何结构队列当前完整训练{status["completed_new"]}/18，仍在运行；仅完整三seed组报告test，未完成组不报最高seed。','',
    '|几何方法|三个seed test宏F1|均值|相对同主干原W0平均差（百分点）|','|---|---|---:|---:|']
for name in ['fov_residual','field_pool','reference','no_field','no_local_norm','no_pool']:
    if name not in field['summary']:continue
    values=field['summary'][name]['seed_macro_f1'];cmp=field['comparisons'].get(name+'_vs_qualification_swin_t_B');diff=cmp['mean_delta'] if cmp else float('nan')
    lines.append('|'+name+'|'+'/'.join(f'{100*v:.4f}%' for v in values)+f'|{100*np.mean(values):.4f}%|{100*diff:+.4f}|')
lines +=['','几何完整机制尚未确认可靠收益，现有阴性和全部参照保留，不因其失败否定所有新设计。CPU患者×类别原型诊断没有稳定收益，不包装成已有效模块。该诊断v1仅bootstrap返回的患者数元数据误写41，v2用独立混淆计数确认实际39患者抽样、原数值区间不变，旧版另存。','',
    '## 低成本可学习方向矩：已实现，尚未训练','',
    '共享预训练stem滤波器构造四个方向响应，以均值/标准差经零初始化1×1投影加回原路径。原型已通过NumPy算术、C4辅助分支性质、随机负对照、有限差分和flat区梯度、RNG与原224/384 logits一致性，以及冻结B0 ConvNeXt-Tiny224的CPU集成。整个网络不声称严格等变。','',
    f'在主要B0上新增{proto["new_parameters"]:,}参数，矩分支模型总参数27,840,963、一次forward逻辑9.11184 GFLOPs，相对B0约+2.271%。Swin384的对应逻辑预算29.19241G。尚未进行正式batch32 CUDA预检或CLE训练，不将CPU通过当性能提升或论文创新。','',
    '下一正式轮直接优化原冻结B0：先做方向矩和成熟C4训练增强对照各3seed、同公开预训练/同60epoch，分别报告单视角和相同C4推理。模块有效后再扩展必要重训消融并增加1–2个互补机制；不靠成熟模块数量、额外预算或最高seed凑创新。','',
    '## 数据与统计边界','',
    '仍为原A–C train2077/182患者、val427/39患者、test576/41患者，患者split、标签、等级、阈值、清单均不改。3595物理缓存不等于训练使用3595图，每epoch2077实际唯一索引有日志证明。test按用户最新指令参与开发；历史test查看、A–C构建的test标签依赖和等级仅为模型一致性代理均披露。条件bootstrap不校正选择偏差，不能称独立泛化确认、全新外部验证或临床标签正确性证明。','',
    '## 复现、权重与后续','',
    '完整源码、配置、预注册、训练日志、逐图预测、审计、负结果和权重保留服务器/home/login/data/cle/auto/GPT。包中包含来源/权重SHA，权重二进制仍在各原weights目录，不覆盖历史checkpoint。C4单图使用和复现命令见ORIENTATION_USE_ZH.md；结构原型和正式下一轮说明见FORMAL_ORIENTATION_NEXT_ZH.md。低成本新网络未训练，当前没有可宣称有效的新网络权重。',
    '','研究继续，不能仅因成熟C4达到条件数值门槛就把整体目标标为完成。']
(S/'STAGE_REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
receipt=dict(primary_B0_unchanged=True,numerical_fixed_set_gate=gate['numerical_fixed_cohort_gate_passed'],new_network_innovation_completed=False,
    independent_confirmation=False,completed_field_models=status['completed_new'],field_total=18,source_report_sha256=sha(S/'STAGE_REPORT_ZH.md'),
    source_results={str(p.relative_to(R)):sha(p) for p in [C4/'orientation_results.json',C4/'numerical_gate_independent_audit.json',BC/'controls/B0/control_results.json',BC/'controls/DINO/control_results.json',E/'patient_evidence_results.json',P/'numeric_preflight_cpu_tiny224.json',F/'round_results.json']})
(S/'stage_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
sys.path.insert(0,str(E));import patient_evidence as guard
guard.S=S;guard.commit('report: deliver conditional milestone with matched C4 controls, all negatives and unfinished low-cost network objective')
exports=S/'exports';exports.mkdir(exist_ok=True);manifest=[];archive=exports/'new_window_research_evidence.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for name,folder in paths.items():
        for p in sorted(folder.rglob('*')):
            if not p.is_file() or any(part in ('exports','__pycache__') for part in p.relative_to(folder).parts) or p.suffix in ('.lock','.tmp'):continue
            rel=p.relative_to(folder)
            if folder==F and rel.parts[0]=='runs' and not (folder/'runs'/rel.parts[1]/'complete.json').exists():continue
            if folder==F and rel.parts[0]=='logs' and not (folder/'runs'/p.stem/'complete.json').exists():continue
            data=p.read_bytes();target=name+'/'+str(rel);info=tarfile.TarInfo(target);info.size=len(data);info.uid=info.gid=1000;info.uname=info.gname='login';tar.addfile(info,io.BytesIO(data));manifest.append(dict(path=target,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    for name in ['STAGE_REPORT_ZH.md','stage_receipt.json']:
        data=(S/name).read_bytes();info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
    data=json.dumps(dict(snapshot=receipt,files=manifest,ongoing_field_round=True,active_unfinished_field_run_logs_excluded=True,
        trained_weight_binaries='Preserved at registered server weights paths; not copied into archive',feature_arrays_included=True),indent=2,ensure_ascii=False).encode();info=tarfile.TarInfo('EVIDENCE_MANIFEST.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
h=sha(archive);(exports/(archive.name+'.sha256')).write_text(h+'  '+archive.name+'\n');print(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=h,files=len(manifest)),flush=True)
