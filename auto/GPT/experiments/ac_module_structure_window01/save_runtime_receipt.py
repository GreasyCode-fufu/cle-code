"""Retain measured admission and actual new neural-training handoff evidence."""
from pathlib import Path
import csv,json,sys,time
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
cfg=m.verify();post=m.READ(S/'postprocessing_revision_v2.json');assert all(m.sha(S/n)==h for n,h in post['sources'].items())
selected=m.READ(S/'profile/selected.json');measurements={arm:m.READ(S/'profile'/arm/'measurements.json') for arm in m.ARMS};identity=m.READ(S/'manager_entry_identity_v2.json');proc=Path('/proc')/str(identity['pid'])/'cmdline';assert proc.exists() and b'run_module_manager_v2.py' in proc.read_bytes()
status=m.READ(S/'status.json');assert not status.get('failures');completed=[]
for job in m.READ(S/'jobs.json'):
    if m.done(job):
        c=m.READ(S/'runs'/job['id']/'complete.json');completed.append(dict(job=job['id'],seed=job['seed'],epochs=60,checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256'],best_val_epoch=c['best_epoch']))
receipt=dict(timestamp=m.core.timestamp(),manager_identity=identity,manager_live=True,status=status,actual_complete_trainings=completed,measured_admission=selected,measurements=measurements,baseline_config_sha256=m.sha(S/'config.json'),postprocessing_config_sha256=m.sha(S/'postprocessing_revision_v2.json'),field_round_audit_sha256=m.sha(m.F/'independent_audit.json'),scope='Actual training and throughput evidence only; paired3seed efficacy follows each complete group',overall_research_complete=False)
m.SAVE(S/'NEURAL_START_RECEIPT.json',receipt)
lines=['# 新模块轮：实际开训与资源调度证据','',f'快照时间：{receipt["timestamp"]}。原几何轮18模型已完整审计；本轮由实际存活的run_module_manager_v2.py管理，训练CLI和原配置保持冻结。','', '|方法|1并发样本/秒|2并发总样本/秒|4并发总样本/秒|选择并发|','|---|---:|---:|---:|---:|']
for arm,rows in measurements.items():
    v={r['slots']:r['aggregate_samples_per_second'] for r in rows};lines.append('|'+arm+'|'+'|'.join(f'{v[k]:.2f}' if k in v else '未测' for k in [1,2,4])+f'|{selected["slots_by_arm"][arm]}|')
lines+=['','各类均已实测1/2/4并发；较高并发未提供超过所选方案的有效总吞吐增益，因此按预注册“最佳97%的最少并发”规则选择1，不继承旧6/8并发上限。显存留6GiB；原profile逐秒GPU/显存/CPU总量及各核最大值/iowait/RAM、每任务加载等待与吞吐全部保存。benchmark基于真实允许train索引，但短期warmup训练不当成正式60epoch效果。','', f'快照时正式完成{len(completed)}/18个训练任务；活跃任务见NEURAL_START_RECEIPT.json。所有正式epoch仍为同一2077个唯一AC train索引。不能凭首个seed宣布有效或筛最高seed；完整3seed登记后统一复算val/test及固定C4对照。','', '首个WTConv seed45实际完成权重已通过predict_module_model.py固定首张val图单图使用核验，误差见usage_smoke/independent_cli_audit.json；不挑预测正确病例，不是独立泛化/临床确认。','', '旧几何轮完整曲线、所有阴性结论及新模块研究均持续保留。整体任务未完成，后续有效机制保留后再开展互补设计和重训消融，不保证SCI创新或录用。']
(S/'RESOURCE_AND_NEURAL_START_ZH.md').write_text('\n'.join(lines)+'\n')
# Avoid staging alongside this same manager's near-completion or group-audit window.
while True:
    st=m.READ(S/'status.json');safe=bool(st.get('running'))
    for j in st.get('running',[]):
        p=Path('/proc')/str(j['pid'])/'cmdline';log=S/'runs'/j['job']/'train_log.csv';meta=log.parent/'run_meta.json';rows=list(csv.DictReader(log.open())) if log.exists() else []
        safe &= p.exists() and j['job'].encode() in p.read_bytes() and len(rows)<50 and meta.exists() and time.time()-meta.stat().st_mtime>10
    if safe or (S/'independent_audit.json').exists():break
    if st.get('failures'):raise RuntimeError('precise manager failure needs inspection')
    time.sleep(5)
m.commit('perf: retain measured1/2/4 concurrency and verified start of the real paired-module neural study')
print(json.dumps(dict(completed=len(completed),source_config_sha256=receipt['baseline_config_sha256'],manager_pid=identity['pid'])))
