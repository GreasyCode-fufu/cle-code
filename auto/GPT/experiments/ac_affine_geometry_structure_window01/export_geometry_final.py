"""CPU-only completed geometry evidence export, independent telemetry reconstruction."""
from pathlib import Path
from fractions import Fraction
import csv,datetime,hashlib,io,json,os,re,sys,tarfile,time
R=Path('/home/login/data/cle/auto/GPT').resolve();G=R/'experiments/ac_affine_geometry_structure_window01';S=R/'experiments/ac_geometry_final_report_window01';N=R/'experiments/ac_patient_alignment_structure_window01';S.mkdir(exist_ok=True);sys.path.insert(0,str(G));import run_module_structure as m
assert os.getuid()==1000;m.verify();assert m.READ(G/'independent_audit.json')['passed'] and m.READ(G/'direct_control_finisher_complete.json')['passed']
def save(n,x):m.SAVE(S/n,x)
def quiet_commit(roots,message):
 import subprocess
 allowed=[str(p.relative_to(R)) for p in roots]
 while True:
  staged=subprocess.check_output([m.core.GIT,'-C',str(R),'diff','--cached','--name-only'],text=True).splitlines();safe=all(any(p.startswith(a+'/') for a in allowed) for p in staged)
  if (N/'manager_entry_identity_v2.json').exists() and not (N/'independent_audit.json').exists():
   pid=m.READ(N/'manager_entry_identity_v2.json')['pid'];proc=Path('/proc')/str(pid)/'cmdline'
   if proc.exists() and b'run_patient_manager.py' in proc.read_bytes():
    st=m.READ(N/'status.json') if (N/'status.json').exists() else {};safe &= bool(st.get('running'))
    for item in st.get('running',[]):
     log=N/'runs'/item['job']/'train_log.csv';meta=log.parent/'run_meta.json';rows=list(csv.DictReader(log.open())) if log.exists() else [];safe &= len(rows)<43 and meta.exists() and time.time()-meta.stat().st_mtime>10
  if safe:break
  time.sleep(2)
 subprocess.run([m.core.GIT,'-C',str(R),'add','--',*allowed],check=True)
 if subprocess.run([m.core.GIT,'-C',str(R),'diff','--cached','--quiet']).returncode:subprocess.run([m.core.GIT,'-C',str(R),'commit','-m',message],check=True)
jobs=m.READ(G/'jobs.json');job_by_id={j['id']:j for j in jobs};weights=[];logs_by_arm={}
for job in jobs:
 assert m.done(job);folder=G/'runs'/job['id'];comp=m.READ(folder/'complete.json');rows=list(csv.DictReader((folder/'train_log.csv').open()));assert len(rows)==60 and comp['best_epoch']==max(range(60),key=lambda k:float(rows[k]['val_macro_f1']))
 assert m.sha(R/comp['checkpoint'])==comp['checkpoint_sha256'];assert all(int(r['actual_train_count'])==int(r['unique_train_count'])==2077 for r in rows)
 weights.append(dict(job=job['id'],arm=job['arm'],seed=job['seed'],checkpoint=comp['checkpoint'],checkpoint_sha256=comp['checkpoint_sha256'],signature=comp['signature'],best_val_epoch=comp['best_epoch'],train_log_sha256=m.sha(folder/'train_log.csv')))
 start=datetime.datetime.fromisoformat(comp['created_at']).timestamp();end=start+float(rows[-1]['elapsed_seconds']);logs_by_arm.setdefault(job['arm'],[]).append(dict(job=job['id'],train_seconds=sum(float(r['train_seconds']) for r in rows),loader_wait_seconds=sum(float(r['loader_wait_seconds']) for r in rows),compute_seconds=sum(float(r['compute_seconds']) for r in rows),metadata_start_unix=start,metadata_endpoint_unix=end,samples=60*2077,peak_gpu_mib=max(float(r['peak_gpu_mib']) for r in rows)))
save('weights_and_tasks.json',dict(models=12,units=weights,all_actual_weights_sha_verified=True,earliest_val_best=True))
receipt=m.READ(G/'runtime_sampler_complete_v2.json');rawpath=G/'continuous_runtime_process_io.jsonl';csvpath=G/'continuous_runtime_resources.csv'
assert m.sha(rawpath)==receipt['process_io_sha256'] and m.sha(csvpath)==receipt['resources_sha256']
details=[json.loads(line) for line in rawpath.read_text().splitlines()];plain=list(csv.DictReader(csvpath.open()));assert len(details)==len(plain)==receipt['rows']
launches={int(pid):job for job,pid in re.findall(r'^launch (geometry_\S+) (\d+)$',(G/'bootstrap.log').read_text(),re.M)};assert len(launches)==12
manager=m.READ(G/'manager_entry_identity_v2.json')['pid'];counts=[];declared_mismatch=0
for d,row in zip(details,plain):
 assert float(row['unix'])==d['unix'] and float(row['gpu_util'])==d['gpu_util']
 parents=[p for p in d['own_processes'] if 'role' not in p]
 for p in parents:assert p['pid'] in launches and p['job']==launches[p['pid']] and p['job'] in job_by_id and p['ppid']==manager
 for p in d['own_processes']:
  if 'role' in p:assert p['ppid'] in {q['pid'] for q in parents}
 assert len({p['pid'] for p in parents})==len(parents);counts.append(len(parents));declared_mismatch+=int(d['own_workers']!=len(parents))
 assert len({job_by_id[p['job']]['arm'] for p in parents})<=1
 d['verified_worker_count']=len(parents);d['verified_arm']=job_by_id[parents[0]['job']]['arm'] if parents else 'manager_or_transition'
durations=[b['unix']-a['unix'] for a,b in zip(details,details[1:])];assert all(v>0 for v in durations);coverage=details[-1]['unix']-details[0]['unix'];assert abs(sum(durations)-coverage)<1e-8
def aggregate(indices):
 total=sum(durations[i] for i in indices)
 if not total:return dict(seconds=0)
 def mean(k):return sum(details[i][k]*durations[i] for i in indices)/total
 return dict(seconds=total,mean_gpu_util=mean('gpu_util'),fraction_gpu_at_least80=sum(durations[i] for i in indices if details[i]['gpu_util']>=80)/total,mean_cpu_percent=mean('cpu_percent'),mean_max_core_percent=mean('cpu_max_core_percent'),mean_iowait_percent=mean('cpu_iowait_percent'),mean_ram_used_percent=mean('ram_used_percent'),max_gpu_used_mib=max(details[i]['gpu_used_mib'] for i in indices))
allindices=list(range(len(durations)));by_count={str(n):aggregate([i for i in allindices if counts[i]==n]) for n in sorted(set(counts))};by_arm={arm:aggregate([i for i in allindices if details[i]['verified_arm']==arm]) for arm in [*m.ARMS,'manager_or_transition']};assert abs(sum(v['seconds'] for v in by_count.values())-coverage)<1e-8
epoch_stats={}
for arm,v in logs_by_arm.items():
 seconds=sum(x['train_seconds'] for x in v);wait=sum(x['loader_wait_seconds'] for x in v);cohort=max(x['metadata_endpoint_unix'] for x in v)-min(x['metadata_start_unix'] for x in v)
 epoch_stats[arm]=dict(training_epoch_samples_per_second=sum(x['samples'] for x in v)/seconds,loader_wait_fraction_of_training_epoch=wait/seconds,metadata_aligned_cohort_samples_per_second=sum(x['samples'] for x in v)/cohort,metadata_aligned_cohort_seconds=cohort,per_job=v)
resource=dict(passed=True,samples=len(details),first_timestamp=details[0]['timestamp'],last_timestamp=details[-1]['timestamp'],coverage_seconds=coverage,max_sample_gap_seconds=max(durations),verified_max_concurrent_training_parents=max(counts),all12_parent_pid_job_ppid_matches=True,declared_status_count_mismatch_rows=declared_mismatch,all_intervals_partitioned=True,whole_sampler=aggregate(allindices),by_verified_parent_count=by_count,by_arm=by_arm,epoch_log_statistics=epoch_stats,profile_policy=m.READ(G/'profile/selected.json'),resource_sha256=m.sha(csvpath),process_io_sha256=m.sha(rawpath),limitations=['parent count0 may include manager GPU inference, patient bootstrap, transition; not automatically idle','time-weighting uses left sample held until next sample and includes startup/teardown; last instantaneous sample has zero duration','first CPU sample defaults0 before deltas available','sampler covers only recorded first-last interval; no backfilled telemetry after terminal','epoch SPS excludes validation/load/startup; metadata cohort includes validation/checkpoints but excludes prior model construction/load; not exact process-end-to-end SPS','profile cap4 for two arms, but only3 formal seeds actually queued; verified maximum3','approval reviewer quota outage03:08-04:17 prevented agent SSH continuation, normal authorized retry04:21 succeeded; G trained autonomously and terminated03:51; subsequent empty valid queue/experiment preparation is not represented as GPU full'],overall_research_complete=False)
save('independent_runtime_audit.json',resource)
result=m.READ(G/'round_results.json');direct=m.READ(G/'direct_control_comparisons.json');budgets={b['arm']:b for b in m.READ(G/'preflight.json')['budgets']};exact=[]
for arm in m.ARMS:
 for mode in ['single','c4']:
  v=result['summary'][arm+'_'+mode];scores=[]
  for unit in v['seeds']:
   cm=unit['confusion_matrix'];score=sum((Fraction(2*cm[k][k],sum(cm[k])+sum(row[k] for row in cm)) for k in range(3)),Fraction())/3;assert abs(float(score)-unit['macro_f1'])<1e-12;scores.append(score)
  assert abs(float(sum(scores)/3)-v['macro_f1'])<1e-12 and not result['comparisons'][arm+'_'+mode+'_vs_qualification_convnext_tiny_A']['numerical_gate'];exact.append(dict(method=arm+'_'+mode,seed_exact_fractions=list(map(str,scores)),mean_exact_fraction=str(sum(scores)/3)))
save('independent_summary_fraction_audit.json',dict(passed=True,methods=exact,all8_goal_gates_false=True,not_selection_bias_adjusted=True))
lines=['# CLE局部几何结构完整轮结果','','四方法×三个配对seed共12模型全部完成60epoch，最早最高原0度val macroF1选checkpoint，完整427val/576test两概率路径、训练索引、权重有限值和SHA、患者隔离/login归属独立审计通过。全部8个推理方案未达严格目标，原M方向矩继续保留，局部仿射增量不作为已确认有效组件；整体研究继续，新的患者条件辅助目标已独立冻结并启动预检查/吞吐/训练自动链。','', '|方法/推理|seed45 F1|seed46|seed47|均值|对固定B0差（百分点）|41患者条件95%区间|','|---|---:|---:|---:|---:|---:|---|']
for arm in m.ARMS:
 for mode in ['single','c4']:
  name=arm+'_'+mode;v=result['summary'][name];d=result['comparisons'][name+'_vs_qualification_convnext_tiny_A'];lo,hi=d['conditional_patient_bootstrap']['ci95'];lines.append('|'+name+'|'+'|'.join(f'{100*x:.5f}%' for x in v['seed_macro_f1']+[v['macro_f1']])+f'|{100*d["mean_delta"]:.5f}|[{100*lo:.5f},{100*hi:.5f}]|')
lines+=['','## 机制、去一与成熟参照','','完整mom_affine对原Mom单视图平均差−0.712308百分点，患者条件区间全负（并非每seed均负）；C4平均−0.878153百分点、三个seed均负且区间全负。完整C4对同参数/offset范数的permuted控制平均−0.522130百分点、三个seed均负且区间全负，不支持本实现的连续局部仿射布局带来增益。对自由DCN的小点估计增量区间跨0；去方向矩也未提供稳定互补证据。','', 'permuted C4对固定B0+1.120462百分点且条件区间为正，平均差仍小于1.5；不以局部正向结果或四舍五入宣称成功。原Mom C4的+1.476485百分点同样严格不足。比较使用全部三seed，原M和B0有效权重按签名/SHA复用，未重训削弱参照。全部预注册直接对照见DIRECT_CONTROL_REPORT_ZH.md和direct_control_comparisons.json，整数Fraction独立核验严格3/200门槛。','', '本轮只否定当前放置/约束/训练食谱下的互补假设，不推断所有几何或可变形结构无效。正局部det约束仅适用于打乱前生成的局部基，不保证全局无折叠、临床拓扑或整CNN等变；打乱控制采样格不再仿射，已有独立残差核验。CPU/CUDA数值有效与CLE效果分开记录。','', '## 其他指标与成本','','|方法/推理|accuracy|恶性召回|患者等权accuracy|参数|逻辑GFLOPs|','|---|---:|---:|---:|---:|---:|']
for arm in m.ARMS:
 for mode in ['single','c4']:
  v=result['summary'][arm+'_'+mode];lines.append('|'+arm+'_'+mode+'|'+'|'.join(f'{100*v[k]:.5f}%' for k in ['accuracy','malignant_recall','patient_equal_accuracy'])+f'|{budgets[arm]["parameters"]:,}|{v["counted_flops"]/1e9:.6f}|')
lines+=['','每seed完整指标/混淆矩阵与逐图概率保留在round_results/evaluations；三seed均值不是集成。逻辑2FLOP/MAC含两次DCN收缩但不含插值/归一化/逐元素/访存，实际成本必须同时读吞吐。C4为四倍前向。','', '## 连续资源与实际吞吐','','|实际训练父进程数|覆盖秒|时间加权GPU利用率|GPU≥80%时长占比|CPU总利用率|峰值GPU显存MiB|','|---|---:|---:|---:|---:|---:|']
for n,v in by_count.items():lines.append(f'|{n}|{v["seconds"]:.2f}|{v["mean_gpu_util"]:.2f}%|{100*v["fraction_gpu_at_least80"]:.2f}%|{v["mean_cpu_percent"]:.2f}%|{v["max_gpu_used_mib"]:.0f}|')
lines+=['',f'独立采样{len(details)}行，覆盖{details[0]["timestamp"]}至{details[-1]["timestamp"]}，共{coverage:.2f}秒。实际最多{max(counts)}个训练父进程；两个arm profile cap4但只有三个正式seed，不能称四模型实际并发。JSONL父PID/任务/PPID逐一对应真实启动日志，status计数滞后的{declared_mismatch}行按实际进程还原。时间分区完整和原始CSV/JSONL双SHA通过，启动收尾低GPU记录保留。无训练父进程时可能是manager GPU推理/患者bootstrap/切换，不能一概称空闲。','', '|arm|各任务训练epoch合计吞吐（图/秒）|DataLoader等待/训练时长|元数据对齐三seed队列吞吐|','|---|---:|---:|---:|']
for arm,v in epoch_stats.items():lines.append(f'|{arm}|{v["training_epoch_samples_per_second"]:.2f}|{100*v["loader_wait_fraction_of_training_epoch"]:.3f}%|{v["metadata_aligned_cohort_samples_per_second"]:.2f}|')
lines+=['', '训练epoch吞吐不含val/启动；元数据队列吞吐含val/存checkpoint，起点在模型/loader构造之后，不虚称精确全进程端到端吞吐。每核CPU、iowait、RAM、实际父/子进程I/O与GPU显存原始记录保留，时间加权和采样间隔见independent_runtime_audit.json。','', '03:08自动审批review服务因额度限制未执行SSH，本窗口未绕过；04:21按相同正常权限重试成功，故障已解决。期间G在服务器自主完成；不虚称Agent持续在线监控。G终止到下一有效方案准备期间有真实无任务队列空档，不制造填充训练，不伪造该间隔资源采样或满载。','', '## 复现和使用','','本轮已终态，不恢复旧manager/worker重复训练。源码、配置、预注册、CPU/CUDA检查、并发测量、每epoch索引SHA、日志、逐图概率、直接控制及独立审计全部封存；真实12权重任务/最佳epoch/SHA见weights_and_tasks.json，权重二进制保留服务器。','', '```bash','cd /home/login/data/cle/auto/GPT','/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/ac_affine_geometry_structure_window01/predict_module_model.py --arm mom_affine --seed 45 --image /absolute/path/image.png --views single --output /absolute/path/prediction.json','```','','首val固定index2386真实权重CLI对原batch32概率最大差1.78814e-6<预设1e-4，独立审计通过。模型只加载已完成且SHA一致权重；类别0/1/2，恶性1。结构/预处理/训练入口见原STRUCTURE_AND_USE_ZH.md；新方法并非推荐的临床系统。','', '固定train2077/182、val427/39、test576/41，实际每epoch2077唯一白名单，物理3595缓存不等于训练用3595。所有标签/等级阈值/清单/split保持不变。用户授权test反馈下一设计，历史test已查看且AC构造用其标签；结果属于自适应内部开发，bootstrap只条件描述、不校正选择偏差，不能称独立泛化或新外部验证。等级仅模型一致性代理。成熟DCN/局部仿射已有先验，不自动宣称创新/SCI录用。']
(S/'GEOMETRY_COMPLETE_REPORT_ZH.md').write_text('\n'.join(lines)+'\n');(S/'.gitignore').write_text('exports/\n__pycache__/\n*.pyc\n')
save('final_receipt.json',dict(models=12,source_config_sha256=m.sha(G/'config.json'),full_audit_sha256=m.sha(G/'independent_audit.json'),direct_controls_sha256=m.sha(G/'direct_control_comparisons.json'),all8_goal_gates_false=True,weights_manifest_sha256=m.sha(S/'weights_and_tasks.json'),independent_runtime_sha256=m.sha(S/'independent_runtime_audit.json'),overall_research_complete=False))
quiet_commit([G,S],'results: preserve all12 geometry negatives exact controls verified3 concurrency and resolved quota gap')
data={};manifest=[]
for root in [G,R/'models/optimize/ac_affine_geometry_window01',S]:
 for p in sorted(root.rglob('*')):
  if not p.is_file() or any(x in p.parts for x in ['exports','__pycache__']) or p.suffix in ['.pyc','.lock','.tmp']:continue
  raw=p.read_bytes();name=str(p.relative_to(R));data[name]=raw;manifest.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
out=S/'exports';out.mkdir(exist_ok=True);target=out/'geometry_complete_evidence.tar.gz'
with tarfile.open(target,'w:gz') as tar:
 for name,raw in data.items():
  item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o644;item.uid=item.gid=1000;item.uname=item.gname='login';tar.addfile(item,io.BytesIO(raw))
 raw=json.dumps(dict(files=manifest,all12_models_complete=True,weights_on_server_by_sha=True),ensure_ascii=False,indent=2).encode();item=tarfile.TarInfo('EVIDENCE_MANIFEST.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
m.SAVE(out/'export_receipt.json',dict(path=str(target.relative_to(R)),sha256=m.sha(target),bytes=target.stat().st_size,files=len(manifest)));print(json.dumps(m.READ(out/'export_receipt.json')),flush=True)
