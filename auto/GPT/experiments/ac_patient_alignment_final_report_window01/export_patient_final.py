"""Wait for all registered audits, then preserve full patient-objective round, CPU only."""
from pathlib import Path
from fractions import Fraction
import csv,datetime,hashlib,io,json,os,re,subprocess,sys,tarfile,time,traceback
R=Path('/home/login/data/cle/auto/GPT').resolve();C=R/'experiments/ac_patient_alignment_structure_window01';S=R/'experiments/ac_patient_alignment_final_report_window01';S.mkdir(exist_ok=True);sys.path.insert(0,str(C));import run_module_structure as m
def save(n,x):m.SAVE(S/n,x)
def alive(pid,expected):
 p=Path('/proc')/str(pid)/'cmdline'
 return p.exists() and expected.encode() in p.read_bytes()
def safe_commit():
 allowed=[str(p.relative_to(R)) for p in [C,S]]
 while True:
  staged=subprocess.check_output([m.core.GIT,'-C',str(R),'diff','--cached','--name-only'],text=True).splitlines()
  if all(any(p.startswith(a+'/') for a in allowed) for p in staged):break
  time.sleep(2)
 subprocess.run([m.core.GIT,'-C',str(R),'add','--',*allowed],check=True)
 if subprocess.run([m.core.GIT,'-C',str(R),'diff','--cached','--quiet']).returncode:subprocess.run([m.core.GIT,'-C',str(R),'commit','-m','results: preserve complete patient-objective round all seeds exact controls projection audit and runtime'],check=True)
def main():
 assert os.getuid()==1000 and os.environ.get('CUDA_VISIBLE_DEVICES')=='';m.verify()
 save('exporter_identity.json',dict(pid=os.getpid(),source_sha256=m.sha(__file__),registered_at=m.core.timestamp(),source_config_sha256=m.sha(C/'config.json'),CPU_only=True,new_training=0))
 required=['bootstrap_complete.json','independent_audit.json','direct_control_finisher_complete.json','independent_objective_logging_audit.json','runtime_sampler_complete_v2.json']
 while not all((C/n).exists() for n in required):
  failures=[n for n in ['bootstrap_failure.json','direct_control_finisher_failure.json','objective_logging_finisher_failure.json'] if (C/n).exists()]
  assert not failures,failures
  time.sleep(5)
 for n in required[:-1]:assert m.READ(C/n)['passed'],n
 for identity,expected in [('bootstrap_identity.json','bootstrap_module_structure.py'),('manager_entry_identity_v2.json','run_patient_manager.py'),('direct_control_finisher_identity.json','finish_direct_controls.py'),('objective_logging_finisher_identity.json','audit_objective_logging.py')]:
  while alive(m.READ(C/identity)['pid'],expected):time.sleep(2)
 m.verify();jobs=m.READ(C/'jobs.json');jobmap={j['id']:j for j in jobs};weights=[];epoch_stats={}
 for j in jobs:
  assert m.done(j);folder=C/'runs'/j['id'];comp=m.READ(folder/'complete.json');rows=list(csv.DictReader((folder/'train_log.csv').open()));assert len(rows)==60 and comp['best_epoch']==max(range(60),key=lambda k:float(rows[k]['val_macro_f1'])) and m.sha(R/comp['checkpoint'])==comp['checkpoint_sha256']
  weights.append(dict(job=j['id'],arm=j['arm'],seed=j['seed'],checkpoint=comp['checkpoint'],checkpoint_sha256=comp['checkpoint_sha256'],signature=comp['signature'],best_val_epoch=comp['best_epoch'],train_log_sha256=m.sha(folder/'train_log.csv')))
  start=datetime.datetime.fromisoformat(comp['created_at']).timestamp();epoch_stats.setdefault(j['arm'],[]).append(dict(job=j['id'],train_seconds=sum(float(r['train_seconds']) for r in rows),loader_wait_seconds=sum(float(r['loader_wait_seconds']) for r in rows),start_unix=start,endpoint_unix=start+float(rows[-1]['elapsed_seconds']),samples=60*2077))
 save('weights_and_tasks.json',dict(passed=True,models=12,units=weights,all_actual_weight_sha_verified=True,earliest_val_best=True))
 telemetry=m.READ(C/'runtime_sampler_complete_v2.json');resource=C/'continuous_runtime_resources.csv';process=C/'continuous_runtime_process_io.jsonl';assert m.sha(resource)==telemetry['resources_sha256'] and m.sha(process)==telemetry['process_io_sha256'];rows=[json.loads(v) for v in process.read_text().splitlines()];csvrows=list(csv.DictReader(resource.open()));assert len(rows)==len(csvrows)==telemetry['rows']
 launches={int(pid):job for job,pid in re.findall(r'^launch (patient_\S+) (\d+)$',(C/'bootstrap.log').read_text(),re.M)};assert len(launches)==12;manager=m.READ(C/'manager_entry_identity_v2.json')['pid'];mismatches=0
 for row,flat in zip(rows,csvrows):
  assert row['unix']==float(flat['unix']) and row['gpu_util']==float(flat['gpu_util']);parents=[p for p in row['own_processes'] if 'role' not in p]
  for p in parents:assert p['pid'] in launches and p['job']==launches[p['pid']] and p['ppid']==manager and p['job'] in jobmap
  for p in row['own_processes']:
   if 'role' in p:assert p['ppid'] in {q['pid'] for q in parents}
  row['actual_workers']=len(parents);mismatches+=int(row['own_workers']!=len(parents));assert len({jobmap[p['job']]['arm'] for p in parents})<=1;row['arm']=jobmap[parents[0]['job']]['arm'] if parents else 'manager_or_transition'
 duration=[b['unix']-a['unix'] for a,b in zip(rows,rows[1:])];assert all(v>0 for v in duration);coverage=rows[-1]['unix']-rows[0]['unix'];assert abs(sum(duration)-coverage)<1e-8
 def aggregate(indices):
  total=sum(duration[i] for i in indices)
  if not total:return dict(seconds=0)
  def mean(k):return sum(rows[i][k]*duration[i] for i in indices)/total
  return dict(seconds=total,mean_gpu_util=mean('gpu_util'),fraction_gpu_at_least80=sum(duration[i] for i in indices if rows[i]['gpu_util']>=80)/total,mean_cpu_percent=mean('cpu_percent'),mean_max_core_percent=mean('cpu_max_core_percent'),mean_iowait_percent=mean('cpu_iowait_percent'),mean_ram_used_percent=mean('ram_used_percent'),peak_gpu_used_mib=max(rows[i]['gpu_used_mib'] for i in indices))
 indices=list(range(len(duration)));by_count={str(n):aggregate([i for i in indices if rows[i]['actual_workers']==n]) for n in sorted({r['actual_workers'] for r in rows})};by_arm={a:aggregate([i for i in indices if rows[i]['arm']==a]) for a in [*m.ARMS,'manager_or_transition']};assert abs(sum(v['seconds'] for v in by_count.values())-coverage)<1e-8
 logs={}
 for arm,v in epoch_stats.items():
  seconds=sum(x['train_seconds'] for x in v);span=max(x['endpoint_unix'] for x in v)-min(x['start_unix'] for x in v);logs[arm]=dict(epoch_training_samples_per_second=sum(x['samples'] for x in v)/seconds,loader_wait_fraction=sum(x['loader_wait_seconds'] for x in v)/seconds,metadata_cohort_samples_per_second=sum(x['samples'] for x in v)/span,metadata_cohort_seconds=span,units=v)
 save('independent_runtime_audit.json',dict(passed=True,samples=len(rows),first_timestamp=rows[0]['timestamp'],last_timestamp=rows[-1]['timestamp'],coverage_seconds=coverage,max_gap_seconds=max(duration),whole=aggregate(indices),by_actual_parent_count=by_count,by_arm=by_arm,verified_max_concurrent_training_parents=max(r['actual_workers'] for r in rows),all12_parent_pid_job_ppid_matched=True,all_intervals_partitioned=True,status_count_mismatch_rows=mismatches,profile_policy=m.READ(C/'profile/selected.json'),epoch_log_statistics=logs,resource_sha256=m.sha(resource),process_io_sha256=m.sha(process),scope=['left sample held until next sample; includes startup/teardown, no omitted lowGPU intervals','zero actual worker may mean manager GPU inference/patient bootstrap/transition, not all idle','first CPU sample defaults0','epoch throughput excludes validation/startup; metadata cohort includes validation/checkpoint but excludes model/loader construction, not exact process end-to-end','no telemetry invented outside actual sampler coverage'],overall_research_complete=False))
 result=m.READ(C/'round_results.json');direct=m.READ(C/'direct_control_comparisons.json');objective=m.READ(C/'independent_objective_logging_audit.json');budgets={b['arm']:b for b in m.READ(C/'module_source_audit.json')['budgets']};exact=[];gates=[]
 def fraction(cm):return sum((Fraction(2*cm[k][k],sum(cm[k])+sum(row[k] for row in cm)) for k in range(3)),Fraction())/3
 baseline=[fraction(v['confusion_matrix']) for v in result['summary']['qualification_convnext_tiny_A']['seeds']]
 for arm in m.ARMS:
  for mode in ['single','c4']:
   name=arm+'_'+mode;v=result['summary'][name];scores=[fraction(z['confusion_matrix']) for z in v['seeds']];assert all(abs(float(x)-z['macro_f1'])<1e-12 for x,z in zip(scores,v['seeds']));deltas=[x-y for x,y in zip(scores,baseline)];mean=sum(deltas,Fraction())/3;c=result['comparisons'][name+'_vs_qualification_convnext_tiny_A'];ci=c['conditional_patient_bootstrap']['ci95'];gate=bool(mean>Fraction(3,200) and all(x>0 for x in deltas) and ci[0]>0);assert gate==c['numerical_gate'] and abs(float(mean)-c['mean_delta'])<1e-12
   if gate:gates.append(name)
   exact.append(dict(method=name,seed_fractions=list(map(str,scores)),seed_delta_fractions=list(map(str,deltas)),exact_mean_delta=str(mean),strict_mean_above_3over200=mean>Fraction(3,200),all_seeds_positive=all(x>0 for x in deltas),conditional_ci95=ci,numerical_gate=gate))
 save('independent_summary_fraction_audit.json',dict(passed=True,methods=exact,new_numerical_gate_methods=gates,scope='adaptive internal development conditional numerical gate; not independent generalization or novelty confirmation',overall_research_complete=False))
 lines=['# CLE患者条件表征：完整轮研究报告','','四arm×三个配对seed共12模型全部完成60epoch，原0度val最早最高macroF1选择权重。完整427val/576test两路线概率、训练实际2077索引、有限权重与SHA、患者隔离/login归属、全部直接对照及辅助日志审计通过。全部seed/阴性结果保留，固定B0和有效原Mom参考不削弱不重训。整体网络研究继续，不以本轮结束替代全部目标。','', '新数值门槛成立的方案：'+(', '.join(gates) if gates else '无')+'。门槛由整数混淆矩阵Fraction均值严格>3/200、三seed正、41患者条件区间下界正共同核验；这不是独立泛化或SCI新颖性结论。','', '|方法/推理|seed45 F1|seed46|seed47|均值|对B0差（百分点）|条件95%区间（百分点）|','|---|---:|---:|---:|---:|---:|---|']
 for arm in m.ARMS:
  for mode in ['single','c4']:
   name=arm+'_'+mode;v=result['summary'][name];d=result['comparisons'][name+'_vs_qualification_convnext_tiny_A'];lo,hi=d['conditional_patient_bootstrap']['ci95'];lines.append('|'+name+'|'+'|'.join(f'{100*x:.5f}%' for x in v['seed_macro_f1']+[v['macro_f1']])+f'|{100*d["mean_delta"]:.5f}|[{100*lo:.5f},{100*hi:.5f}]|')
 lines+=['','## 机制与成熟反机制对照','','保留原方向矩，检验同类不同患者对齐是否减少病例记忆，并以去方向矩、成熟标准SupCon、同患者同类的相反机制比较。本实现无额外临床元数据/特殊P×n采样，不称完整PaCL复现，患者对比与投影均有成熟先验。','', '|完整mom_cross对照|三seed差（百分点）|平均差|患者条件95%区间|','|---|---|---:|---|']
 for name,v in direct['comparisons'].items():
  lo,hi=v['conditional_patient_bootstrap']['ci95'];lines.append('|'+name+'|'+','.join(f'{100*x:.5f}' for x in v['paired_seed_deltas'])+f'|{100*v["mean_delta"]:.5f}|[{100*lo:.5f},{100*hi:.5f}]|')
 lines+=['', 'single与C4分别对同推理预算原Mom/成熟参照进行比较。基于本轮证据，候选机制只有在对原Mom提供稳定增量、成熟与反机制/去一对照支持后才考虑保留；不能仅凭对B0小正差或最佳seed把辅助loss包装成创新。原Mom C4对B0+1.476485百分点严格不足1.5，不四舍五入改写门槛。','', '## 其他指标和计算成本','','|方法/推理|accuracy|恶性召回|患者等权accuracy|总训练参数|推理前向参数|逻辑GFLOPs|','|---|---:|---:|---:|---:|---:|---:|']
 for arm in m.ARMS:
  for mode in ['single','c4']:
   v=result['summary'][arm+'_'+mode];b=budgets[arm];lines.append('|'+arm+'_'+mode+'|'+'|'.join(f'{100*v[k]:.5f}%' for k in ['accuracy','malignant_recall','patient_equal_accuracy'])+f'|{b["parameters"]:,}|{b["inference_parameters"]:,}|{v["counted_flops"]/1e9:.6f}|')
 lines+=['', '每seed混淆矩阵和完整概率保留在round_results/evaluations。训练投影106496参数保存于checkpoint，推理不执行、不需要患者ID。逻辑推理2FLOP/MAC含C4算子，不含归一化/逐元素/访存；训练另有投影和batch相似矩阵成本，实测吞吐包含整个辅助目标。C4四倍前向，三seed均值不是集成。','', '## 实际辅助训练暴露','','|arm/seed|平均有效anchor|平均正对数|最大CE+0.05aux恒等误差|','|---|---:|---:|---:|']
 for u in objective['units']:lines.append(f'|{u["arm"]}/{u["seed"]}|{u["mean_valid_anchor_fraction"]:.6f}|{u["mean_positive_pairs"]:.6f}|{u["max_loss_weight_identity_error"]:.3g}|')
 lines+=['', 'aux目标：同类cross/within被排除对不进入伪负分母；同患者不同标签仍负。空正对anchor0、全anchor均值，原随机batch/单图增强与2077每epoch一次保持不变，特别披露within实际稀疏性。独立CPU重建公开同seed初始化并检查选中checkpoint的两投影权重确有更新、所有状态有限，见independent_objective_logging_audit.json。此验证不等于有效性因果证明。','', '## 连续资源与端到端口径','','|实际训练父进程数|覆盖秒|时间加权GPU|GPU≥80%时长|CPU总利用率|峰值GPU显存MiB|','|---|---:|---:|---:|---:|---:|']
 for n,v in by_count.items():lines.append(f'|{n}|{v["seconds"]:.2f}|{v["mean_gpu_util"]:.2f}%|{100*v["fraction_gpu_at_least80"]:.2f}%|{v["mean_cpu_percent"]:.2f}%|{v["peak_gpu_used_mib"]:.0f}|')
 lines+=['',f'采样{len(rows)}行，{rows[0]["timestamp"]}至{rows[-1]["timestamp"]}，覆盖{coverage:.2f}秒。实际父PID/任务/PPID与12次启动映射全通过，status数滞后{mismatches}行按真实进程恢复；不把0训练父进程下的manager推理/患者分析都称空闲。启动收尾低GPU不删，无记录时段不补造。CPU每核/单核、iowait、RAM、进程/磁盘I/O、显存及采样间隙见原始JSONL/CSV和independent_runtime_audit.json。','', '|arm|训练epoch累计图/秒|DataLoader等待/训练时长|元数据队列对齐图/秒|','|---|---:|---:|---:|']
 for arm,v in logs.items():lines.append(f'|{arm}|{v["epoch_training_samples_per_second"]:.2f}|{100*v["loader_wait_fraction"]:.3f}%|{v["metadata_cohort_samples_per_second"]:.2f}|')
 lines+=['', '训练epoch吞吐不含val/启动；队列对齐吞吐含val/存checkpoint，起点在模型/loader构造之后，不称精确全进程端到端值。所有四完整目标先测单/并发，GPU满载且无吞吐收益不继续扩容，不为填GPU制造无意义训练。','', '## 复现和模型使用','','本轮已完成，不恢复旧bootstrap/manager重复训练；所有源码/配置/注册/CPU/CUDA预检/性能/日志/逐图预测/独立审计封存。12个真实checkpoint签名、最佳epoch与SHA见weights_and_tasks.json，权重二进制保留服务器。','','```bash','cd /home/login/data/cle/auto/GPT','/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/ac_patient_alignment_structure_window01/predict_module_model.py --arm mom_cross --seed 45 --image /absolute/path/image.png --views single --output /absolute/path/prediction.json','```','','固定首val2386真实CLI对batch32概率误差1.603621e-6<预设1e-4，通过usage_smoke审计；使用脚本拒绝未完成/SHA不符模型。类别0/1/2，恶性1。原食谱60epoch/batch32、encoder1e-5、新头/方向矩/投影1e-4、AdamW.05/cosine0/clip1/AMP、HV/rotation15/worker2/thread1；温度.1、aux权重.05、单视图固定，结构细节和原始训练入口见STRUCTURE_AND_USE_ZH.md。','', '## 边界与继续依据','','固定A–C train2077/182、val427/39、test576/41，3595物理cache严格白名单访问，各epoch2077唯一索引SHA审计。标签、等级、阈值、患者split和筛选清单不改变。用户授权test反馈设计，历史test被查看且AC构建用过标签，所有test为自适应内部开发；bootstrap仅条件描述，不校正选择偏差，不称独立泛化或全新外部验证。等级仅模型一致性代理。结果不保证SCI创新、可靠外部改善或录用；依累计/去一/成熟反机制结果决定下一方向，不把普通阴性作为整体停止条件。']
 (S/'PATIENT_ALIGNMENT_COMPLETE_REPORT_ZH.md').write_text('\n'.join(lines)+'\n');save('final_receipt.json',dict(passed=True,models=12,source_config_sha256=m.sha(C/'config.json'),full_audit_sha256=m.sha(C/'independent_audit.json'),direct_controls_sha256=m.sha(C/'direct_control_comparisons.json'),objective_audit_sha256=m.sha(C/'independent_objective_logging_audit.json'),weights_manifest_sha256=m.sha(S/'weights_and_tasks.json'),new_numerical_gate_methods=gates,independent_generalization_confirmation=False,overall_research_complete=False));safe_commit()
 data={};manifest=[]
 for root in [C,m.M,S]:
  for p in sorted(root.rglob('*')):
   if not p.is_file() or any(x in p.parts for x in ['exports','__pycache__']) or p.suffix in ['.pyc','.lock','.tmp']:continue
   raw=p.read_bytes();name=str(p.relative_to(R));data[name]=raw;manifest.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
 out=S/'exports';out.mkdir(exist_ok=True);target=out/'patient_alignment_complete_evidence.tar.gz'
 with tarfile.open(target,'w:gz') as tar:
  for name,raw in data.items():
   item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o644;item.uid=item.gid=1000;item.uname=item.gname='login';tar.addfile(item,io.BytesIO(raw))
  raw=json.dumps(dict(files=manifest,all12_models_complete=True,weights_on_server_by_sha=True),ensure_ascii=False,indent=2).encode();item=tarfile.TarInfo('EVIDENCE_MANIFEST.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 m.SAVE(out/'export_receipt.json',dict(path=str(target.relative_to(R)),sha256=m.sha(target),bytes=target.stat().st_size,files=len(manifest)));print(json.dumps(m.READ(out/'export_receipt.json')),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save('export_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
