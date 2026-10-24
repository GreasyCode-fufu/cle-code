"""Wait for all registered audits, then preserve full power-shape round, CPU only."""
from pathlib import Path
from fractions import Fraction
import csv,datetime,hashlib,io,json,os,re,subprocess,sys,tarfile,time,traceback
R=Path('/home/login/data/cle/auto/GPT').resolve();C=R/'experiments/ac_harmonic_partition_structure_window01';S=R/'experiments/ac_power_shape_final_report_window01';S.mkdir(exist_ok=True);sys.path.insert(0,str(C));import run_module_structure as m
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
 if subprocess.run([m.core.GIT,'-C',str(R),'diff','--cached','--quiet']).returncode:subprocess.run([m.core.GIT,'-C',str(R),'commit','-m','results: preserve complete power-shape round all seeds exact controls mechanism controls and runtime'],check=True)
def main():
 assert os.getuid()==1000 and os.environ.get('CUDA_VISIBLE_DEVICES')=='';m.verify()
 save('exporter_identity.json',dict(pid=os.getpid(),source_sha256=m.sha(__file__),registered_at=m.core.timestamp(),source_config_sha256=m.sha(C/'config.json'),CPU_only=True,new_training=0))
 required=['bootstrap_complete.json','independent_audit.json','direct_control_finisher_complete.json','runtime_sampler_complete_v2.json']
 while not all((C/n).exists() for n in required):
  failures=[n for n in ['bootstrap_failure.json','direct_control_finisher_failure.json'] if (C/n).exists()]
  assert not failures,failures
  time.sleep(5)
 for n in required[:-1]:assert m.READ(C/n)['passed'],n
 for identity,expected in [('bootstrap_identity.json','bootstrap_module_structure.py'),('manager_entry_identity_v2.json','run_power_manager.py'),('direct_control_finisher_identity.json','finish_direct_controls.py')]:
  while alive(m.READ(C/identity)['pid'],expected):time.sleep(2)
 m.verify();jobs=m.READ(C/'jobs.json');jobmap={j['id']:j for j in jobs};weights=[];epoch_stats={}
 for j in jobs:
  assert m.done(j);folder=C/'runs'/j['id'];comp=m.READ(folder/'complete.json');rows=list(csv.DictReader((folder/'train_log.csv').open()));assert len(rows)==60 and comp['best_epoch']==max(range(60),key=lambda k:float(rows[k]['val_macro_f1'])) and m.sha(R/comp['checkpoint'])==comp['checkpoint_sha256']
  weights.append(dict(job=j['id'],arm=j['arm'],seed=j['seed'],checkpoint=comp['checkpoint'],checkpoint_sha256=comp['checkpoint_sha256'],signature=comp['signature'],best_val_epoch=comp['best_epoch'],train_log_sha256=m.sha(folder/'train_log.csv')))
  start=datetime.datetime.fromisoformat(comp['created_at']).timestamp();epoch_stats.setdefault(j['arm'],[]).append(dict(job=j['id'],train_seconds=sum(float(r['train_seconds']) for r in rows),loader_wait_seconds=sum(float(r['loader_wait_seconds']) for r in rows),start_unix=start,endpoint_unix=start+float(rows[-1]['elapsed_seconds']),samples=60*2077))
 save('weights_and_tasks.json',dict(passed=True,models=15,units=weights,all_actual_weight_sha_verified=True,earliest_val_best=True))
 telemetry=m.READ(C/'runtime_sampler_complete_v2.json');resource=C/'continuous_runtime_resources.csv';process=C/'continuous_runtime_process_io.jsonl';assert m.sha(resource)==telemetry['resources_sha256'] and m.sha(process)==telemetry['process_io_sha256'];rows=[json.loads(v) for v in process.read_text().splitlines()];csvrows=list(csv.DictReader(resource.open()));assert len(rows)==len(csvrows)==telemetry['rows']
 launches={int(pid):job for job,pid in re.findall(r'^launch (power_\S+) (\d+)$',(C/'bootstrap.log').read_text(),re.M)};assert len(launches)==15;manager=m.READ(C/'manager_entry_identity_v2.json')['pid'];mismatches=0
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
 save('independent_runtime_audit.json',dict(passed=True,samples=len(rows),first_timestamp=rows[0]['timestamp'],last_timestamp=rows[-1]['timestamp'],coverage_seconds=coverage,max_gap_seconds=max(duration),whole=aggregate(indices),by_actual_parent_count=by_count,by_arm=by_arm,verified_max_concurrent_training_parents=max(r['actual_workers'] for r in rows),all15_parent_pid_job_ppid_matched=True,all_intervals_partitioned=True,status_count_mismatch_rows=mismatches,profile_policy=m.READ(C/'profile/selected.json'),epoch_log_statistics=logs,resource_sha256=m.sha(resource),process_io_sha256=m.sha(process),scope=['left sample held until next sample; includes startup/teardown, no omitted lowGPU intervals','zero actual worker may mean manager GPU inference/patient bootstrap/transition, not all idle','first CPU sample defaults0','epoch throughput excludes validation/startup; metadata cohort includes validation/checkpoint but excludes model/loader construction, not exact process end-to-end','no telemetry invented outside actual sampler coverage'],overall_research_complete=False))
 result=m.READ(C/'round_results.json');direct=m.READ(C/'direct_control_comparisons.json');budgets={b['arm']:b for b in m.READ(C/'module_source_audit.json')['budgets']};exact=[];gates=[]
 def fraction(cm):return sum((Fraction(2*cm[k][k],sum(cm[k])+sum(row[k] for row in cm)) for k in range(3)),Fraction())/3
 baseline=[fraction(v['confusion_matrix']) for v in result['summary']['qualification_convnext_tiny_A']['seeds']]
 for arm in m.ARMS:
  for mode in ['single','c4']:
   name=arm+'_'+mode;v=result['summary'][name];scores=[fraction(z['confusion_matrix']) for z in v['seeds']];assert all(abs(float(x)-z['macro_f1'])<1e-12 for x,z in zip(scores,v['seeds']));deltas=[x-y for x,y in zip(scores,baseline)];mean=sum(deltas,Fraction())/3;c=result['comparisons'][name+'_vs_qualification_convnext_tiny_A'];ci=c['conditional_patient_bootstrap']['ci95'];gate=bool(mean>Fraction(3,200) and all(x>0 for x in deltas) and ci[0]>0);assert gate==c['numerical_gate'] and abs(float(mean)-c['mean_delta'])<1e-12
   if gate:gates.append(name)
   exact.append(dict(method=name,seed_fractions=list(map(str,scores)),seed_delta_fractions=list(map(str,deltas)),exact_mean_delta=str(mean),strict_mean_above_3over200=mean>Fraction(3,200),all_seeds_positive=all(x>0 for x in deltas),conditional_ci95=ci,numerical_gate=gate))
 save('independent_summary_fraction_audit.json',dict(passed=True,methods=exact,new_numerical_gate_methods=gates,scope='adaptive internal development conditional numerical gate; not independent generalization or novelty confirmation',overall_research_complete=False))
 lines=['# CLE C4功率形状：完整轮结果','','五方法×三配对seed共15模型全部60epoch，原0度val最早最高macroF1选checkpoint。完整427val/576test两路线概率、2077唯一训练索引、有限权重与source/task/checkpointSHA、患者隔离/login归属、预注册直接控制均独立审计通过。全部seed与阴性保留，固定B0/原Mom有效权重按签名复用，不削弱参照。整体研究继续，不以本轮结束代替全部目标。','', '新的条件数值门槛方案：'+(', '.join(gates) if gates else '无')+'。独立整数Fraction均值严格>3/200、三seed正且41患者条件bootstrap下界正同时核验；不把该门槛称独立泛化或SCI新颖性确认。','', '|方法/推理|seed45 F1|seed46|seed47|均值|对B0差（百分点）|41患者条件95%区间|','|---|---:|---:|---:|---:|---:|---|']
 for arm in m.ARMS:
  for mode in ['single','c4']:
   name=arm+'_'+mode;v=result['summary'][name];d=result['comparisons'][name+'_vs_qualification_convnext_tiny_A'];lo,hi=d['conditional_patient_bootstrap']['ci95'];lines.append('|'+name+'|'+'|'.join(f'{100*x:.5f}%' for x in v['seed_macro_f1']+[v['macro_f1']])+f'|{100*d["mean_delta"]:.5f}|[{100*lo:.5f},{100*hi:.5f}]|')
 lines+=['','## 机制、累计与去一','','原Mom的µ/σ保留，检验其丢失的方向周期能量分配是否提供增量。E1=((r0−r2)²+(r1−r3)²)/8，E2=(r0−r1+r2−r3)²/16，var=E1+E2，q=(E2−E1)/(E1+E2+1e−6)。µ/σ相同而q不同的数学反例并不能证明CLE效果，须依去一/错循环/重复信息/成熟幅值控制解释。','', '|mom_shape比较|三seed差（百分点）|平均差|条件95%区间|','|---|---|---:|---|']
 for name,v in direct['comparisons'].items():
  lo,hi=v['conditional_patient_bootstrap']['ci95'];lines.append('|'+name+'|'+','.join(f'{100*x:.5f}' for x in v['paired_seed_deltas'])+f'|{100*v["mean_delta"]:.5f}|[{100*lo:.5f},{100*hi:.5f}]|')
 lines+=['', '对原Mom是去q增量；shape_only真实不建立18528参数Mom适配器；permuted同响应边际/额外参数但固定非循环排列；duplicate额外分支重复σ以控制优化重参数化；rawh2为成熟DFT第二频率幅值。不能仅按对B0的最好seed或额外参数引起的小正差宣称机制成立。','', '完整mom_shape相对原Mom：']
 for mode in ['single','c4']:
  v=direct['comparisons'][mode+'_vs_M_moments'];lo,hi=v['conditional_patient_bootstrap']['ci95'];direction='三seed均正' if all(x>0 for x in v['paired_seed_deltas']) else '三seed均负' if all(x<0 for x in v['paired_seed_deltas']) else '跨seed方向不一致或含零';interval='下界为正' if lo>0 else '上界为负' if hi<0 else '跨0';lines.append(f'- {mode}：平均{100*v["mean_delta"]:.5f}百分点，{direction}，条件区间{interval}。')
 lines+=['', '只有对原Mom有稳定增量且其余成熟/去一对照支持，才考虑保留新组件；一般阴性继续据机制调整，不否定全部傅里叶/群方法。原Mom+C4相对B0+1.476485百分点严格不足1.5，不能舍入成功。','', '## 强参照与推理预算','','|新方法/推理|对B0同C4差（百分点）|条件95%区间|对DINO同C4差|条件95%区间|','|---|---:|---|---:|---|']
 for arm in m.ARMS:
  name=arm+'_c4';b=result['comparisons'][name+'_vs_B0_C4'];d=result['comparisons'][name+'_vs_DINO_C4'];bl,bh=b['conditional_patient_bootstrap']['ci95'];dl,dh=d['conditional_patient_bootstrap']['ci95'];lines.append(f'|{name}|{100*b["mean_delta"]:.5f}|[{100*bl:.5f},{100*bh:.5f}]|{100*d["mean_delta"]:.5f}|[{100*dl:.5f},{100*dh:.5f}]|')
 lines+=['', '同C4列避免把成熟四视图收益归给新增结构；不同主干预训练/实际成本仍需同时阅读预算。固定B0主比较不事后改变。Swin_C4成熟参考及其它全部比较保留在round_results，历史test选择依赖已披露。','', '## 其他指标、混淆矩阵和成本','','|方法/推理|accuracy|恶性召回|患者等权accuracy|参数|逻辑GFLOPs|','|---|---:|---:|---:|---:|---:|']
 for arm in m.ARMS:
  for mode in ['single','c4']:
   v=result['summary'][arm+'_'+mode];b=budgets[arm];lines.append('|'+arm+'_'+mode+'|'+'|'.join(f'{100*v[k]:.5f}%' for k in ['accuracy','malignant_recall','patient_equal_accuracy'])+f'|{b["parameters"]:,}|{v["counted_flops"]/1e9:.6f}|')
 lines+=['', '每seed完整accuracy/恶性召回/患者等权accuracy与3×3混淆矩阵保留在round_results.json和evaluations/*/s*/independent_audit.json；原始单图和四旋转概率全部入包。三seed均值不是集成。逻辑2FLOP/MAC含3个额外功能旋转卷积/1×1，不含功率/除法/归一化/逐元素/访存；同参数和MAC不意味着相同实际延迟，C4四倍前向，必须合看实测。','', '## 真实资源与耗时','','|实际训练父进程数|覆盖秒|时间加权GPU|GPU≥80%时长|CPU总利用率|峰值显存MiB|','|---|---:|---:|---:|---:|---:|']
 for n,v in by_count.items():lines.append(f'|{n}|{v["seconds"]:.2f}|{v["mean_gpu_util"]:.2f}%|{100*v["fraction_gpu_at_least80"]:.2f}%|{v["mean_cpu_percent"]:.2f}%|{v["peak_gpu_used_mib"]:.0f}|')
 lines+=['', f'独立采样{len(rows)}行，覆盖{rows[0]["timestamp"]}至{rows[-1]["timestamp"]}共{coverage:.2f}秒。15次启动PID/任务/PPID映射全核验，status计数滞后{mismatches}行按真实进程恢复。区间完整分区；启动、收尾和分析低GPU不删，0训练父进程可能是manager GPU推理/患者bootstrap/切换，不都称空闲；无记录时段不补造。CPU每核/单核、iowait/RAM、GPU显存、父/子进程和磁盘I/O原始数据及最大采样间隙全部保留。','', '|arm|训练epoch累计图/秒|加载等待/训练时长|元数据队列对齐图/秒|','|---|---:|---:|---:|']
 for arm,v in logs.items():lines.append(f'|{arm}|{v["epoch_training_samples_per_second"]:.2f}|{100*v["loader_wait_fraction"]:.3f}%|{v["metadata_cohort_samples_per_second"]:.2f}|')
 lines+=['', '训练epoch吞吐排除val/启动；元数据队列口径含val/存checkpoint但从模型/loader构造之后开始，不能称精确全进程端到端吞吐。五内核先独立测1/2/4总吞吐，最佳97%内最小并发皆1，保留6GiB；GPU已满载而吞吐无益不制造无价值并发。','', '## 数值、复现和模型使用','','保留NumPy FFT/Parseval/反例/循环与空间90协变/错序破坏/退化与eps缩放/doublegradcheck、官方H-Net纯幅值AST输出/梯度、公开初始化/原参数ID/RNG证明。正式新增真正去Mom、七公开起点及所有新投影反向，CUDA真实batch32两步有限、新投影梯度非零、六B0/Mom完整427概率误差0.0/cacheSHA/test训练加载拒绝，均在训练前完成。数学/数值证明与CLE效果分开。','', '本轮已终态，不重复启动旧bootstrap/manager/训练。全部source/config/注册/预检/性能/日志/逐图概率/主审计/直接Fraction封存；15实际权重/任务签名/最佳epoch/SHA见weights_and_tasks.json，二进制仍在服务器。','','```bash','cd /home/login/data/cle/auto/GPT','/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/ac_harmonic_partition_structure_window01/predict_module_model.py --arm mom_shape --seed 45 --image /absolute/path/image.png --views single --output /absolute/path/prediction.json','```','','固定首val2386真实权重CLI对batch32概率误差2.8073787689e−5<预设1e−4，usage_smoke已独立通过；不据单图/单seed称有效。RGB→256双线性→中心224→ImageNet归一，类别0/1/2，恶性1，C4四旋转概率均值，拒绝未完成/SHA不符权重。原CE60epoch/batch32、encoder1e−5、新分支/头1e−4、AdamW.05/cosine0/clip1/AMP、HV/rotation15/worker2/thread1，同公开ImageNet和配对45/46/47，无旧checkpoint热启动；详细训练入口见STRUCTURE_AND_USE_ZH。','', '## 边界和后续','','固定A–C train2077/182患者、val427/39、test576/41，物理3595cache仅白名单使用，每epoch2077唯一索引SHA。标签/等级阈值/清单/患者split不变；等级仅模型一致性代理。用户明确test反馈设计，历史test已查看且AC构建用其标签，全部结果为自适应内部开发，条件bootstrap不校正选择偏差，不称独立泛化或新外部验证。','', 'H-Net圆谐波/群傅里叶/幅值均有成熟先验，rawh2只忠实于官方幅值算子并保留MIT源码/blob，不是完整H-Net/连续360网络/ECCV2018整法复现。局部q不保证普通尾部整网等变，eps下缩放非严格不变，不把数学量直接当临床形态诊断；原型初始启动器类型故障与独立修复完整保留，不改冻结算子。成熟组合不自动SCI创新，不保证提点或录用。整体目标active，依据全部控制决定下一机制，未达不能自行停止。']
 (S/'POWER_SHAPE_COMPLETE_REPORT_ZH.md').write_text('\n'.join(lines)+'\n');save('final_receipt.json',dict(passed=True,models=15,source_config_sha256=m.sha(C/'config.json'),full_audit_sha256=m.sha(C/'independent_audit.json'),direct_controls_sha256=m.sha(C/'direct_control_comparisons.json'),weights_manifest_sha256=m.sha(S/'weights_and_tasks.json'),new_numerical_gate_methods=gates,independent_generalization_confirmation=False,overall_research_complete=False));safe_commit()
 data={};manifest=[]
 for root in [C,m.M,S,R/'experiments/ac_harmonic_partition_prototype_window01']:
  for p in sorted(root.rglob('*')):
   if not p.is_file() or any(x in p.parts for x in ['exports','__pycache__']) or p.suffix in ['.pyc','.lock','.tmp']:continue
   raw=p.read_bytes();name=str(p.relative_to(R));data[name]=raw;manifest.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
 out=S/'exports';out.mkdir(exist_ok=True);target=out/'power_shape_complete_evidence.tar.gz'
 with tarfile.open(target,'w:gz') as tar:
  for name,raw in data.items():
   item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o644;item.uid=item.gid=1000;item.uname=item.gname='login';tar.addfile(item,io.BytesIO(raw))
  raw=json.dumps(dict(files=manifest,all15_models_complete=True,weights_on_server_by_sha=True),ensure_ascii=False,indent=2).encode();item=tarfile.TarInfo('EVIDENCE_MANIFEST.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 m.SAVE(out/'export_receipt.json',dict(path=str(target.relative_to(R)),sha256=m.sha(target),bytes=target.stat().st_size,files=len(manifest)));print(json.dumps(m.READ(out/'export_receipt.json')),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save('export_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
