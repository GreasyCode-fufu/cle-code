"""Snapshot actual formal startup evidence; never present incomplete training as efficacy."""
from pathlib import Path
import csv,hashlib,io,json,os,socket,subprocess,sys,tarfile
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import run_field_structure as f

cfg=f.verify();ref=f.READ(S/'working_reference.json');feedback=f.READ(S/'evidence/test_feedback_snapshot.json');q=f.READ(f.Q/'qualification_results.json')
state=f.READ(S/'queue_status_v2.json');identity=f.READ(S/'queue_manager_identity.json');pid=identity['pid']
actual=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'manage_field_queue_v2.py' in actual
progress=[]
for item in state['running']:
    cmd=(Path('/proc')/str(item['pid'])/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert item['job'] in cmd
    log=list(csv.DictReader((S/'runs'/item['job']/'train_log.csv').open()));progress.append(dict(**item,completed_epochs=len(log),actual_command=cmd))
budget=f.READ(S/'preflight.json');b0='qualification_'+cfg['primary_baseline'];w0=ref['method'];threshold=feedback['summary'][b0]['macro_f1']+.015
summary=dict(snapshot_at=f.core.timestamp(),host=socket.gethostname(),uid=os.getuid(),config_sha256=f.sha(S/'config.json'),
    primary_B0=cfg['primary_baseline'],B0_validation_macro_f1=q['summary'][cfg['primary_baseline']]['macro_f1'],B0_test_macro_f1=feedback['summary'][b0]['macro_f1'],
    working_W0=ref['arm'],W0_test_macro_f1=feedback['summary'][w0]['macro_f1'],threshold_strictly_greater_than=threshold,
    formal_preflight_passed=budget['passed'],running=progress,completed_new=state['completed_new'],total_new=18,manager_identity=identity,
    efficacy_established=False,H1_test_evaluated=False,overall_research_complete=False,independent_generalization_confirmed=False)
f.SAVE(S/'startup_snapshot.json',summary)
lines=['# 正式几何视场支持结构研究：启动快照','',f'快照：{summary["snapshot_at"]}。源码、配置、任务与CUDA预检已提交；H1已真实训练。当前{state["completed_new"]}/18模型训练完成，{len(progress)}个原训练进程存活。H1效果尚未成立，不报告其测试收益。','',
    '|参照|选择依据|验证macroF1|test三seed均值|','|---|---|---:|---:|',
    f'|固定B0：{cfg["primary_baseline"]}|预先冻结的公平资格val规则|{100*summary["B0_validation_macro_f1"]:.4f}%|{100*summary["B0_test_macro_f1"]:.4f}%|',
    f'|同主干参照W0：{ref["arm"]}|完整8种资格方案test均值最大，明确test选择|{100*q["summary"][ref["arm"]]["macro_f1"]:.4f}%|{100*summary["W0_test_macro_f1"]:.4f}%|',
    f'|DINO强参照|历史开发选择、保留对照|98.2784%|{100*feedback["summary"]["dino_baseline"]["macro_f1"]:.4f}%|','',
    f'相对已冻结B0，最终平均test宏F1必须严格超过{100*threshold:.6f}%，且所有配对seed差值为正、患者条件bootstrap下界为正。仍须披露test参与开发；这些条件不能转称独立泛化确认。B0不按test降低或替换。','',
    '## H1与预算','',
    '固定单位圆支持复制RGB几何增强；只改变空间stem、深度可分离空间卷积或Swin窗口注意力，以及最终池化。新增分支以零初始化tanh标量门残差加入原输出，RGB切片contiguous。直接池化、直接NVIDIA公式和去视场/归一化/聚合对照全部保留。成熟算子不声称新颖，性能和消融完成后再判断机制。','',
    '|结构|参数|GFLOPs|batch32训练峰值MiB|','|---|---:|---:|---:|']
for b in budget['budgets']:lines.append(f'|{b["arm"]}|{b["parameters"]:,}|{b["counted_flops"]/1e9:.5f}|{b["peak_gpu_mib"]:.2f}|')
lines+=['','计算按2 FLOP/MAC，显式计算卷积/线性/注意力，不包含norm、激活、softmax、pool和逐元素操作。原W0/B0有效训练复用；18个新模型均从对应公开预训练起步、每seed60epoch，原encoder1e-5，新增门与head1e-4。','',
    '正式batch32 CUDA/AMP两步有限梯度、14个完整H1门非零梯度、零门logits逐位相同、三个原W0 checkpoint全427验证概率复算、优化器覆盖/门学习率和test拒绝检查通过。CPU原型的5008 RGB/RNG及160空间对齐证据按完全相同数据/算子源码SHA复用。','',
    '## 实际调度与连续性','',
    '实测完整H1单任务178.12张/秒、GPU均值98.86%；两任务合计165.20张/秒、GPU100%。原队列把启动低利用窗口误当并发增加依据，已经以单独v2调度源码和策略SHA修正；原冻结模型、配方和源签名不变。两项已启动CUDA worker原进程收养并继续，不重训。其后完整H1按实测1任务准入；field_pool/reference计算族另做真实单/并发测量。','',
    '当前状态以queue_status_v2.json、queue_manager_identity.json和/proc实际命令为准。旧bootstrap_status/status/pipeline_pid/launch为历史阶段入口；QUEUE_TRANSITION_V2.json记录主动替换，旧管理PID缺失不代表模型失效。不要按旧launch重启模型。后续若管理器故障，先核对活worker并更新收养登记；complete/checkpoint有效则只补推理。','',
    'continuous_resources_v2.csv每2秒采GPU、显存、CPU总/单核、I/O等待、RAM；io_process_resources.jsonl持续采系统物理磁盘和本流水线进程I/O。每epoch记录加载等待、计算、吞吐、实际2077唯一索引和显存。','',
    '## 审计与后续','',
    '每方法全部3seed结束后先Git登记完整checkpoint SHA，再独立构造全427 val/576 test概率、状态与指标复算，保存全部逐图预测、混淆矩阵、患者配对条件bootstrap及门值。test反馈用来设计下一方法，参数梯度仍只用train，不改标签/等级/阈值/清单。','',
    '正常运行和一般阴性保持安静。H1有效则保留后加1–2个互补机制，累加和drop-one重训；阴性则据证据改机制，不把这一轮完成写成整个目标完成。完整结构图与使用命令见FIELD_STRUCTURE_AND_USE_ZH.md；新调度优先级见本快照。当前不存在可宣布有效的H1最终模型。']
(S/'FORMAL_START_REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
commits=subprocess.check_output([f.GIT,'-C',str(f.R),'log','-8','--format=%h %s','--',str(S.relative_to(f.R))],text=True).strip()
(S/'RESEARCH_LEDGER.md').write_text('# 本窗口正式几何结构研究账本\n\n'+f'当前快照：{summary["snapshot_at"]}。整体目标active，已实施而非建议；所有效应待完整3seed审计。\n\n'+'\n'.join('- '+x for x in commits.splitlines())+'\n\n后续逐方法登记与结果分别保存在evaluations、round_results.json及FIELD_ROUND_REPORT_ZH.md，按Git历史保留阴性。\n',encoding='utf-8')
f.commit('docs: record actual formal training startup, competitive references, budgets and preserved worker continuity')
dest=S/'exports';dest.mkdir(exist_ok=True);archive=dest/'formal_field_startup_evidence.tar.gz';manifest=[]
with tarfile.open(archive,'w:gz') as tar:
    for path in sorted(S.rglob('*')):
        if not path.is_file() or any(part in ('exports','__pycache__') for part in path.relative_to(S).parts) or path.suffix in ('.tmp','.lock'):continue
        data=path.read_bytes();name=str(path.relative_to(S));info=tarfile.TarInfo(name);info.size=len(data);info.uid=info.gid=1000;info.uname=info.gname='login';tar.addfile(info,io.BytesIO(data));manifest.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    payload=json.dumps(dict(snapshot=summary,files=manifest,ongoing_not_final_result=True),indent=2,ensure_ascii=False).encode();info=tarfile.TarInfo('SNAPSHOT_MANIFEST.json');info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
h=hashlib.sha256(archive.read_bytes()).hexdigest();(dest/(archive.name+'.sha256')).write_text(h+'  '+archive.name+'\n')
print(dict(snapshot=summary['snapshot_at'],archive=str(archive),sha256=h,files=len(manifest)))
