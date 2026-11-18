"""Wait for all registered audits, then preserve full Swin merge-contrast round, CPU only."""
from pathlib import Path
from fractions import Fraction
import csv,datetime,hashlib,io,json,os,re,subprocess,sys,tarfile,time,traceback
R=Path('/home/login/data/cle/auto/GPT').resolve();C=R/'experiments/ac_merge_contrast_structure_window01';S=R/'experiments/ac_merge_contrast_final_report_window01';S.mkdir(exist_ok=True);sys.path.insert(0,str(C));import run_module_structure as m
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
 if subprocess.run([m.core.GIT,'-C',str(R),'diff','--cached','--quiet']).returncode:subprocess.run([m.core.GIT,'-C',str(R),'commit','-m','results: preserve complete Swin merge-contrast round all seeds exact controls mechanism controls and runtime'],check=True)
def main():
 assert os.getuid()==1000 and os.environ.get('CUDA_VISIBLE_DEVICES')=='';m.verify()
 save('exporter_identity.json',dict(pid=os.getpid(),source_sha256=m.sha(__file__),registered_at=m.core.timestamp(),source_config_sha256=m.sha(C/'config.json'),CPU_only=True,new_training=0))
 required=['usage_smoke/independent_cli_audit.json','bootstrap_complete.json','independent_audit.json','direct_control_finisher_complete.json','runtime_sampler_complete_v2.json']
 while not all((C/n).exists() for n in required):
  failures=[n for n in ['bootstrap_failure.json','direct_control_finisher_failure.json','usage_smoke_failure.json'] if (C/n).exists()]
  assert not failures,failures
  time.sleep(5)
 for n in required[:-1]:assert m.READ(C/n)['passed'],n
 for identity,expected in [('bootstrap_identity.json','bootstrap_module_structure.py'),('manager_entry_identity_v2.json','run_merge_manager.py'),('direct_control_finisher_identity.json','finish_direct_controls.py')]:
  while alive(m.READ(C/identity)['pid'],expected):time.sleep(2)
 m.verify();jobs=m.READ(C/'jobs.json');jobmap={j['id']:j for j in jobs};weights=[];epoch_stats={}
 for j in jobs:
  assert m.done(j);folder=C/'runs'/j['id'];comp=m.READ(folder/'complete.json');rows=list(csv.DictReader((folder/'train_log.csv').open()));assert len(rows)==60 and comp['best_epoch']==max(range(60),key=lambda k:float(rows[k]['val_macro_f1'])) and m.sha(R/comp['checkpoint'])==comp['checkpoint_sha256']
  weights.append(dict(job=j['id'],arm=j['arm'],seed=j['seed'],checkpoint=comp['checkpoint'],checkpoint_sha256=comp['checkpoint_sha256'],signature=comp['signature'],best_val_epoch=comp['best_epoch'],train_log_sha256=m.sha(folder/'train_log.csv')))
  start=datetime.datetime.fromisoformat(comp['created_at']).timestamp();epoch_stats.setdefault(j['arm'],[]).append(dict(job=j['id'],train_seconds=sum(float(r['train_seconds']) for r in rows),loader_wait_seconds=sum(float(r['loader_wait_seconds']) for r in rows),start_unix=start,endpoint_unix=start+float(rows[-1]['elapsed_seconds']),samples=60*2077))
 save('weights_and_tasks.json',dict(passed=True,models=15,units=weights,all_actual_weight_sha_verified=True,earliest_val_best=True))
 telemetry=m.READ(C/'runtime_sampler_complete_v2.json');resource=C/'continuous_runtime_resources.csv';process=C/'continuous_runtime_process_io.jsonl';assert m.sha(resource)==telemetry['resources_sha256'] and m.sha(process)==telemetry['process_io_sha256'];rows=[json.loads(v) for v in process.read_text().splitlines()];csvrows=list(csv.DictReader(resource.open()));assert len(rows)==len(csvrows)==telemetry['rows']
 launches={int(pid):job for job,pid in re.findall(r'^launch (merge_\S+) (\d+)$',(C/'bootstrap.log').read_text(),re.M)};assert len(launches)==15;manager=m.READ(C/'manager_entry_identity_v2.json')['pid'];mismatches=0
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
 lines=['# CLE Swin384方向矩迁移与首merge局部空间对比：完整轮','','15模型/三seed/60epoch全部保留；原val最早最高macroF1权重，完整427val/576test两路线概率、有限权重/source/job/checkpointSHA/2077唯一索引/患者隔离/所有权及直接整数Fraction控制审计通过。整体目标继续，不把一轮结束当全部完成。','', '新的条件数值门槛方案：'+(', '.join(gates) if gates else '无')+'；门槛不能代替强同预算参照、创新性或独立泛化确认。','', '|方法|s45 F1|s46 F1|s47 F1|均值|对B0差pp|条件95%区间pp|','|---|---:|---:|---:|---:|---:|---|']
 for arm in m.ARMS:
  for mode in ['single','c4']:
   name=arm+'_'+mode;v=result['summary'][name];d=result['comparisons'][name+'_vs_qualification_convnext_tiny_A'];lo,hi=d['conditional_patient_bootstrap']['ci95'];lines.append('|'+name+'|'+'|'.join(f'{100*x:.5f}%' for x in v['seed_macro_f1']+[v['macro_f1']])+f'|{100*d["mean_delta"]:.5f}|[{100*lo:.5f},{100*hi:.5f}]|')
 lines+=['', '## 累计/去一/空间与成熟父模型控制','','mom_only检验原方向矩向Swin384的迁移；mom_local在此基础加first merge局部标准差；local_only去Mom；mom_mean同参数均值；mom_global相同h先空间均值广播移除位置。保留原concat→learnedLN→Linear，新增统计独立noaffineLN eps1e-5，h=sqrt(mean_q((u_q-mean_q u)^2)+1e-6)-.001，零96→192线性残差。原对象/RNG/公开起点保持，未热启动已训练checkpoint。标准差/LN/Haar/残差是成熟先验，组合不自动新颖。','', '|mom_local比较|三seed差pp|平均差pp|条件95%区间pp|','|---|---|---:|---|']
 for name,v in direct['comparisons'].items():
  lo,hi=v['conditional_patient_bootstrap']['ci95'];lines.append('|'+name+'|'+','.join(f'{100*x:.5f}' for x in v['paired_seed_deltas'])+f'|{100*v["mean_delta"]:.5f}|[{100*lo:.5f},{100*hi:.5f}]|')
 lines+=['', 'Swin_B比较为同主干同recipe原父模型，C4为同四视图父模型；B0仍固定Tiny_A224的单视图主比较。工作Swin B由历史test选择，所有结果为自适应内部开发。不能把近阈值、最好seed或成熟C4收益写成新机制成功。','', '|方案|对SwinB单视图差pp|条件区间pp|对Swin同C4差pp|条件区间pp|','|---|---:|---|---:|---|']
 for arm in m.ARMS:
  x=result['comparisons'][arm+'_single_vs_qualification_swin_t_B'];z=result['comparisons'][arm+'_c4_vs_Swin_C4'];a,b=x['conditional_patient_bootstrap']['ci95'];c,d=z['conditional_patient_bootstrap']['ci95'];lines.append(f'|{arm}|{100*x["mean_delta"]:.5f}|[{100*a:.5f},{100*b:.5f}]|{100*z["mean_delta"]:.5f}|[{100*c:.5f},{100*d:.5f}]|')
 lines+=['', '## 指标与预算','','|方法|accuracy|恶性召回|患者等权accuracy|参数|逻辑GFLOPs|','|---|---:|---:|---:|---:|---:|']
 for arm in m.ARMS:
  for mode in ['single','c4']:
   v=result['summary'][arm+'_'+mode];b=budgets[arm];lines.append('|'+arm+'_'+mode+'|'+'|'.join(f'{100*v[k]:.5f}%' for k in ['accuracy','malignant_recall','patient_equal_accuracy'])+f'|{b["parameters"]:,}|{v["counted_flops"]/1e9:.6f}|')
 lines+=['', '每seed3×3整数混淆矩阵和全部指标见round_results及独立audit；三seed均值非集成。原Swin B28.597871616G，Mom29.192414208G，local28.682806272G，累计29.277348864G；固定B0仅8.909531136G，不隐藏主干成本约3.2倍。逻辑2FLOP/MAC含attention/三额外旋转conv/newLinear，排除LN/统计/逐元素/访存。','', '## 实际资源','','|真实父worker数|覆盖秒|时间加权GPU|GPU>=80%时长|CPU总量|峰显存MiB|','|---|---:|---:|---:|---:|---:|']
 for n,v in by_count.items():lines.append(f'|{n}|{v["seconds"]:.2f}|{v["mean_gpu_util"]:.2f}%|{100*v["fraction_gpu_at_least80"]:.2f}%|{v["mean_cpu_percent"]:.2f}%|{v["peak_gpu_used_mib"]:.0f}|')
 lines+=['',f'独立{len(rows)}行覆盖{coverage:.3f}秒；15次真实launch/PID/PPID映射核验，status计数滞后{mismatches}行据实际恢复。启动/收尾/分析低GPU完整保留，0worker可能manager推理或CPU分析，不一概称空闲，无记录不补造。原始CPU每核/进程/IO/RAM和采样间隙在独立runtime报告。各arm只有3正式seed，profilecap不等于真实并发。','', '|arm|训练epoch图/秒|加载等待比例|元数据队列图/秒|','|---|---:|---:|---:|']
 for arm,v in logs.items():lines.append(f'|{arm}|{v["epoch_training_samples_per_second"]:.2f}|{100*v["loader_wait_fraction"]:.3f}%|{v["metadata_cohort_samples_per_second"]:.2f}|')
 usage=m.READ(C/'usage_smoke/independent_cli_audit.json')
 lines+=['', '## 复现和使用','','原Q Swin_t_B公开ImageNet/stochastic depth .1/60epoch/batch32/encoder1e-5/new与head1e-4/AdamW.05/cosine0/clip1/AMP/HVrotation15/worker2/thread1，原图RGB448缓存中心或随机384，非256上采样。val原0度最早最高macroF1选权重，labels/grades/split/采样不变，每epoch2077唯一索引SHA。所有source/config冻结，15实际checkpoint/task/最佳epoch/SHA见weights_and_tasks，权重二进制在服务器。','','```bash','cd /home/login/data/cle/auto/GPT','/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/ac_merge_contrast_structure_window01/predict_module_model.py --arm mom_local --seed 45 --image /absolute/path/image.png --views single --output /absolute/path/prediction.json','```','',f'首真实mom_only45固定val{usage["index"]} CLI对原batch32误差{usage["max_probability_difference_batch1_vs_original_batch32"]}<预设1e-4；只是实际使用核验，无新训练或test图，非三seed效果。类别0/1/2恶性1，C4四旋转概率均值，拒绝未完成或SHA错误。','', '## 科学边界','','原Mom保留但向新主干迁移效果须依本轮控制判断；局部对比度/均值/LN/Haar关系为成熟先验，本实现不是HWD整法复现。原型官方Microsoft/Torchvision PatchMerging偶数输入输出/所有梯度严格对齐、奇数runtimepad原样保留，NumPy/Haar/反例/doublegradcheck/常量有限通过；不据此称CLE有效、整网旋转等变或临床拓扑正确。','', '固定AC train2077/182患者、val427/39、test576/41，恶性1、患者可多标签。用户授权test反馈下一设计；历史test查看且AC构造用标签，工作SwinB选择依赖历史test。所有bootstrap仅41患者条件描述，不校正适应性选择偏差，不称独立泛化或新外部验证。固定B0不替换，成熟SwinC4此前已报告，不能重复归因新结构。不保证提点、SCI新颖或录用；整体active，后续依累计/去一/成熟控制证据持续设计。']
 (S/'MERGE_CONTRAST_COMPLETE_REPORT_ZH.md').write_text('\n'.join(lines)+'\n');save('final_receipt.json',dict(passed=True,models=15,source_config_sha256=m.sha(C/'config.json'),full_audit_sha256=m.sha(C/'independent_audit.json'),direct_controls_sha256=m.sha(C/'direct_control_comparisons.json'),weights_manifest_sha256=m.sha(S/'weights_and_tasks.json'),new_numerical_gate_methods=gates,independent_generalization_confirmation=False,overall_research_complete=False));safe_commit()
 data={};manifest=[]
 for root in [C,m.M,S,R/'experiments/ac_merge_contrast_prototype_window01']:
  for p in sorted(root.rglob('*')):
   if not p.is_file() or any(x in p.parts for x in ['exports','__pycache__']) or p.suffix in ['.pyc','.lock','.tmp']:continue
   raw=p.read_bytes();name=str(p.relative_to(R));data[name]=raw;manifest.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
 out=S/'exports';out.mkdir(exist_ok=True);target=out/'merge_contrast_complete_evidence.tar.gz'
 with tarfile.open(target,'w:gz') as tar:
  for name,raw in data.items():
   item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o644;item.uid=item.gid=1000;item.uname=item.gname='login';tar.addfile(item,io.BytesIO(raw))
  raw=json.dumps(dict(files=manifest,all15_models_complete=True,weights_on_server_by_sha=True),ensure_ascii=False,indent=2).encode();item=tarfile.TarInfo('EVIDENCE_MANIFEST.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 m.SAVE(out/'export_receipt.json',dict(path=str(target.relative_to(R)),sha256=m.sha(target),bytes=target.stat().st_size,files=len(manifest)));print(json.dumps(m.READ(out/'export_receipt.json')),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save('export_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
