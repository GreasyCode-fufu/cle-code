"""Actual-start receipt and immutable snapshot, no new neural work."""
from pathlib import Path
import csv,hashlib,io,json,os,sys,tarfile,time,traceback
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
def main():
 m.verify();assert os.getuid()==1000
 while True:
  assert not (S/'bootstrap_failure.json').exists(),'inspect exact own preflight/profile fault, no training restart'
  if (S/'status.json').exists():
   status=m.READ(S/'status.json');ready=bool(status.get('running'))
   for item in status.get('running',[]):
    p=S/'runs'/item['job']/'train_log.csv';ready &= p.exists() and len(p.read_text().splitlines())>1
   if ready:break
  time.sleep(2)
 workers=[]
 for item in status['running']:
  proc=Path('/proc')/str(item['pid']);assert proc.stat().st_uid==1000 and item['job'].encode() in (proc/'cmdline').read_bytes();rows=list(csv.DictReader((S/'runs'/item['job']/'train_log.csv').open()));assert all(int(r['actual_train_count'])==int(r['unique_train_count'])==2077 for r in rows);workers.append(dict(job=item['job'],pid=item['pid'],epochs=len(rows),actual_train_count=2077,index_set_sha256=rows[-1]['train_index_set_sha256'],samples_per_second=float(rows[-1]['train_samples_per_second'])))
 pre=m.READ(S/'preflight.json');cpu=m.READ(S/'module_source_audit.json');assert pre['passed'] and cpu['passed'];policy=m.READ(S/'profile/selected.json');profiles={p.parent.name:m.READ(p) for p in (S/'profile').glob('*/measurements.json')};resource=list(csv.DictReader((S/'continuous_runtime_resources.csv').open()));recent=resource[-60:]
 receipt=dict(created_at=m.core.timestamp(),source_config_sha256=m.sha(S/'config.json'),CPU_audit_sha256=m.sha(S/'module_source_audit.json'),CUDA_audit_sha256=m.sha(S/'preflight.json'),comparison_preregistration_sha256=m.sha(S/'comparison_preregistration.json'),models=15,completed=status['completed_new'],verified_live_workers=workers,profile_policy=policy,mean_recent_GPU=sum(float(r['gpu_util']) for r in recent)/len(recent),recent_resource_rows=len(recent),real_training_started=True,three_seed_effectiveness_not_established=True,overall_research_complete=False);m.SAVE(S/'NEURAL_START_RECEIPT.json',receipt)
 lines=['# CLE C4功率形状：实际开训、结构与使用','','患者条件表征12模型已终态且全部8方案未达严格目标，未确认辅助对比目标优于原Mom。完整307文件包已交付并本地全SHA核验，原Mom及CE食谱保留。新轮五机制×三seed共15模型先冻结、CPU/CUDA预检、实测并发后实际开训。本文件为启动快照，不能当CLE提点结论。','',f'快照{receipt["created_at"]}；config SHA {receipt["source_config_sha256"]}。','', '## 结构与机制','','```mermaid','flowchart TD',' I[RGB CLE224] --> R[原4×4 stem权重旋转四次得到r0..r3]',' R --> B[保留原r0响应]',' R --> M[保留原均值µ和标准差σ]',' M --> P[原Mom零初始化1×1投影]',' R --> E[方向DFT功率E1 E2]',' E --> Q[q为E2减E1除以总功率加eps]',' Q --> A[新增96→96零初始化1×1]',' B --> S[三分支相加]',' P --> S',' A --> S',' S --> T[原ConvNeXt其余层与分类头]','```','','µ/σ不能唯一确定E1/E2分配，已有合成反例/NumPy FFT/Parseval证明；这只是新增描述信息的数学证据。q在C4循环移位下不变，低能量有限，强度缩放因eps仅近似不变。普通网络尾部不受群约束，不能称整个CNN等变/不变，不能从频谱量直接推导临床组织结构。','', '|arm|机制|参数|单视图逻辑GFLOPs|','|---|---|---:|---:|']
 names=dict(mom_shape='保留µσ并加稳定功率对比q',shape_only='真正去掉18528参数Mom分支',mom_permuted='同响应边际/参数，固定非循环排列破坏邻接',mom_duplicate='同新分支参数但重复原σ',mom_rawh2='成熟原DFT第二频率幅值算子')
 for b in cpu['budgets']:
  if b['arm'] in m.ARMS:lines.append(f'|{b["arm"]}|{names[b["arm"]]}|{b["parameters"]:,}|{b["counted_flops"]/1e9:.6f}|')
 lines+=['', '额外分支9312参数、复用原四响应，保留原base/Mom分支对象与公开初始化RNG；shape_only不存在unused Mom adapter。原Mom为去q参考，五arm分别检验增量、去一、方向关系、重参数化和成熟幅值。逻辑2FLOP/MAC包括功能旋转卷积/模块1×1，排除功率/除法/逐元素/归一化/访存，不能当相同实际延迟；各arm独立实测。C4四倍前向，三seed均值不是集成。','', '## 数值与成熟参照','','原型CPU通过NumPy FFT/Parseval、同µσ不同q、四循环/输入90°局部空间协变和错序破坏、低能量/常量有限、eps伸缩偏差、double gradcheck。官方H-Net纯幅值函数AST以受限NumPy/Torch shim核对输出/输入梯度，官方blob和MIT许可保留；不安装旧TensorFlow，不称完整H-Net/连续旋转网络或ECCV2018整法复现。共享FMA空间rfft2与方向轴变换不同。','', '原型最初启动器将字符串__file__传给Path-only SHA函数，在CPU检查前报错。原source/config/failure保留，独立修复运行器只补启动与登记，算子/测试不改；数值证明在修复后实际完成。正式CPU新增真正去Mom检查、七种公开起点初始logit/参数ID/RNG及有限梯度全部通过；真实batch32各arm两次AMP更新有限，新投影梯度非零，六B0/Mom完整427val概率复建最大差均0.0；物理cacheSHA和test训练加载拒绝通过。数值检查不能当CLE有效性。','', '## 全目标实际并发吞吐','','|arm|1并发图/秒|2并发总图/秒|4并发总图/秒|97%策略cap|','|---|---:|---:|---:|---:|']
 for arm in m.ARMS:
  values={v['slots']:v['aggregate_samples_per_second'] for v in profiles[arm]};lines.append('|'+arm+'|'+'|'.join(f'{values[k]:.2f}' if k in values else '未继续' for k in [1,2,4])+f'|{policy["slots_by_arm"][arm]}|')
 lines+=['', '按最佳实测97%内最小并发、6GiB余量选择；GPU满载且吞吐无益时不继续扩容。每arm仅3正式seed，cap4不能说实际四模型并行。原始profile和正式每2秒GPU/显存、CPU总量/每核、iowait、RAM、自身父/子进程与I/O持续采样，逐epoch等待/耗时/吞吐/2077索引SHA保留。启动快照短时平均GPU '+f'{receipt["mean_recent_GPU"]:.2f}%'+'仅指已采集的'+str(len(recent))+'行，不能替代完整长时统计。','', '## 训练、复现和使用','','固定train2077/182患者、val427/39、test576/41，3595物理cache仅按允许A–C索引使用。固定B0不改，不更换标签/等级/阈值/split/筛图；原CE、60epoch/batch32、encoder1e−5、新分支/头1e−4、AdamW.05/cosine0/clip1/AMP、HV/rotation15/worker2/thread1。seed45/46/47从同公开ImageNet权重和配对头独立训练，无旧checkpoint热启动。原0度val最早最高macroF1选权重，完整三seed组再两路线独立val/test概率审计。当前管理器已经运行，不能重复启动bootstrap。', '', '真实checkpoint完成后使用：','','```bash','cd /home/login/data/cle/auto/GPT','/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/ac_harmonic_partition_structure_window01/predict_module_model.py --arm mom_shape --seed 45 --image /absolute/path/image.png --views single --output /absolute/path/prediction.json','```','','脚本拒绝未完成或SHA不符权重，RGB→256双线性→中心224→ImageNet归一；类别0/1/2，恶性1，C4四旋转概率均值。权重weights/ac_harmonic_partition_structure_window01/power_<arm>_s<seed>/model_best.pth，任务签名/SHA见complete.json。首个真实完成模型之后再固定首val图做CLI smoke，当前不将尚未执行的检查写成已通过。','', '## 评价和边界','','所有15模型全部seed/阴性保留，报告macroF1、accuracy、恶性召回、患者等权accuracy、混淆矩阵和计算成本。直接比较完整mom_shape对原Mom、去Mom、错循环、重复σ、成熟幅值、固定B0，两推理预算分列。41患者20k配对bootstrap仅条件区间，独立整数Fraction门槛平均严格>0.015、三seed正且区间下界正；另核强同预算参照，不挑最高seed。','', '用户授权test反馈设计，历史test已被查看且AC构造用过标签，所有结果属于自适应内部开发，不校正选择偏差，不称独立泛化/全新外部验证。等级仅模型一致性代理。群/傅里叶/幅值均有成熟先验，组合不自动称SCI创新；不保证提点或录用，整体研究继续。']
 (S/'STRUCTURE_AND_USE_ZH.md').write_text('\n'.join(lines)+'\n');(S/'RESEARCH_LEDGER_ZH.md').write_text('# C4功率形状轮账本\n\nP全12模型/60epoch/主概率/直接Fraction/目标投影审计通过且无新数值达标，完整307包已交付；原Mom保留，aux不作为确认增量。H原型预注册/原始hash故障/独立修复/证明均留存，新正式15模型在CPU/CUDA通过后实测并发并实际开训。\n\n'+json.dumps(receipt,ensure_ascii=False,indent=2)+'\n\n整体目标active，不能因本轮或阴性自停。\n')
 while True:
  st=m.READ(S/'status.json');safe=bool(st.get('running'))
  for item in st.get('running',[]):
   p=S/'runs'/item['job']/'train_log.csv';meta=p.parent/'run_meta.json';logs=list(csv.DictReader(p.open())) if p.exists() else [];safe &= len(logs)<43 and meta.exists() and time.time()-meta.stat().st_mtime>10
  if safe or (S/'independent_audit.json').exists():break
  time.sleep(2)
 m.commit('docs: preserve actual15-model power-shape neural start structure compute and reproducible usage')
 H=m.R/'experiments/ac_harmonic_partition_prototype_window01';data={};manifest=[]
 for root in [S,m.M,H]:
  for p in sorted(root.rglob('*')):
   if not p.is_file() or any(x in p.parts for x in ['exports','__pycache__','runs','logs','evaluations']) or p.suffix in ['.pyc','.lock','.tmp'] or p.name in ['continuous_runtime_resources.csv','continuous_runtime_process_io.jsonl','resources.csv','io_resources.jsonl']:continue
   raw=p.read_bytes();name=str(p.relative_to(m.R));data[name]=raw;manifest.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
 out=S/'exports';out.mkdir(exist_ok=True);target=out/'power_shape_start_evidence.tar.gz'
 with tarfile.open(target,'w:gz') as tar:
  for name,raw in data.items():
   item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o644;item.uid=item.gid=1000;item.uname=item.gname='login';tar.addfile(item,io.BytesIO(raw))
  raw=json.dumps(dict(files=manifest,start_snapshot=True,not_complete_round=True,prototype_failure_and_repair_included=True),ensure_ascii=False,indent=2).encode();item=tarfile.TarInfo('EVIDENCE_MANIFEST.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 m.SAVE(out/'export_receipt.json',dict(path=str(target.relative_to(m.R)),sha256=m.sha(target),bytes=target.stat().st_size,files=len(manifest)));print(json.dumps(m.READ(out/'export_receipt.json')),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:m.SAVE(S/'document_start_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
