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
 lines=['# CLE Swin方向矩迁移与局部空间对比：实际开训/复现/使用','','POWER全15模型已经完整独立概率与直接Fraction审计，10方案均未达；392文件完整包本地逐SHA通过。q/rawH2/重复σ无确认增量，原方向矩和CE保留。新轮五arm×三seed共15模型，正式CPU/CUDA和原参考完整427概率重建通过后实测并发并开训，不能据数值预检称效果。','',f'实际快照 {receipt["created_at"]}，配置SHA {receipt["source_config_sha256"]}。','', '## 结构','','```mermaid','flowchart TD',' X[RGB448中心或随机384] --> SM[原Swin stem加原方向矩零投影]',' SM --> B[原前两block]',' B --> Q[原2x2四象限拼接]',' Q --> L[原学习LayerNorm和Linear保留]',' Q --> N[独立无仿射LayerNorm eps1e-5]',' N --> H[按通道的四象限均值与局部标准差]',' H --> A[新增零初始化96到192投影]',' A --> S[残差相加]',' L --> S',' S --> T[原Swin其余stage与分类头]','```','','h=sqrt(mean_q((u_q-mean_q u)^2)+1e-6)-.001；不带仿射LN避免各象限gamma/beta破坏统计置换性质，原learnedLN继续原主支。平坦h严格0，eps下强度缩放近似；不保证普通Swin整网旋转等变/临床拓扑。标准统计/LN/残差和Haar关系已有成熟先验，组合不自动SCI创新。','', '|arm|参数|逻辑GFLOPs|','|---|---:|---:|']
 for b in cpu['budgets']:lines.append(f'|{b["arm"]}|{b["parameters"]:,}|{b["counted_flops"]/1e9:.6f}|')
 lines+=['', 'mom_only是原Mom的Swin迁移去local参考；mom_local累计；local_only真去Mom；mom_mean同参数均值控制；mom_global相同h先空间均值后广播，移除局部位置。额外merge18624参数，原Mom18528，原对象/参数/RNG/公开logit保持，没有已训练权重热启动。Swin384成本约29.3G，固定B0仅8.91G，不隐瞒差异；还要看同Swin父模型和同C4参照。逻辑2FLOP/MAC含Swin功能attention/三旋转卷积/新Linear，排除LN/统计/逐元素/访存，不当真实耗时。','', '## 训练和数据','','固定AC train2077/182患者、val427/39、test576/41，患者可多标签、恶性1，标签/等级/阈值/清单/split/采样不变。每epoch2077唯一索引SHA，B缓存原图RGB resize448/2504图仅train+val，独立T test576×448，均原图SHA来源核验，绝非256缓存上采样。原Q公开ImageNet、stochastic depth .1、60epoch/batch32、encoder1e-5/新增模块与head1e-4、AdamW.05/cosine0/clip1/AMP、HV/rotation15/randomcrop384、workers2/thread1。原SwinB三seed有效权重复用，固定B0不改；工作SwinB基于历史test选择需披露。','', '原0度val最早最高macroF1选权重，三seed完整后才注册并测single/C4，完整427val/576test的直接像素路线与独立T TestData B加载路线审计；权重有限/source/job/checkpointSHA/整数混淆矩阵与41患者20k条件bootstrap。对mom_local预注册mom_only/local_only/mom_mean/mom_global/SwinB/B0两路线直接控制。test为用户授权自适应内部开发，历史查看/AC构建用标签，不称独立泛化，bootstrap不校正选择偏差。','', '## 数值与成熟参照','','原型NumPy pad/四象限/LN/方差和独立正交Haar功率oracle、同均值不同局部对比、local/global区别、偶数90度描述协变/奇数右下pad原样保留、常量低能量有限、doublegradcheck、两合成更新通过。官方Microsoft PatchMerging受限AST和安装Torchvision在偶数输入输出/输入及所有参数梯度严格一致；原码拒绝奇数，runtime支持，不能混称。保留官方源/blob和MIT及Torchvision BSD许可；没有声称HWD整法复现。正式6种公共Swin384初始logit/RNG/原参数ID/有限反向与预算通过，真实batch32全部两次更新、新Mom/merge投影首步梯度非零，三SwinB+三B0完整427概率误差全部0，训练加载拒绝test。','', '## 实测吞吐与使用','','|arm|1并发图秒|2并发总图秒|4并发总图秒|97%策略cap|','|---|---:|---:|---:|---:|']
 for arm in m.ARMS:
  vals={v['slots']:v['aggregate_samples_per_second'] for v in profiles[arm]};lines.append('|'+arm+'|'+'|'.join(f'{vals[k]:.2f}' if k in vals else '未继续' for k in [1,2,4])+f'|{policy["slots_by_arm"][arm]}|')
 lines+=['', '并发基于实际吞吐/显存/CPU，保留6GiB，最佳97%内最小选择；GPU满载无益不扩容，每arm仅3正式seed不能把cap4称四正式并发。每2秒正式GPU/显存/CPU每核/总量/iowait/RAM、自身父子进程和磁盘I/O加逐epoch等待/吞吐保留。这里GPU '+f'{receipt["mean_recent_GPU"]:.2f}%'+'仅启动快照'+str(len(recent))+'行，完整时间加权留最终轮，不删启动/收尾低GPU。','', '管理器已运行，不重复启动。已完成有效权重后：','','```bash','cd /home/login/data/cle/auto/GPT','/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/ac_merge_contrast_structure_window01/predict_module_model.py --arm mom_local --seed 45 --image /absolute/path/image.png --views single --output /absolute/path/prediction.json','```','','脚本拒绝未完成/SHA不符权重，RGB原图448双线性/中心384/ImageNet归一、类别0/1/2恶性1，C4四旋转概率均值非三seed集成；首真实模型固定首val图CLI对原batch32检查另写usage_smoke，尚未通过前不宣称已通过。权重 weights/ac_merge_contrast_structure_window01/merge_<arm>_s<seed>/model_best.pth。所有seed/阴性保留，不保证提点、SCI新颖或录用。数值门槛必须独立Fractionmean>3/200、三seed正、条件CI正并核强同预算参照；整体目标继续。']
 (S/'STRUCTURE_AND_USE_ZH.md').write_text('\n'.join(lines)+'\n');(S/'RESEARCH_LEDGER_ZH.md').write_text('# Swin384迁移与首merge局部统计启动账本\n\n'+json.dumps(receipt,ensure_ascii=False,indent=2)+'\n\nPOWER已终态完整交付；有效原Mom保留，未确认q/患者aux不带入。本轮启动快照非完整15模型效果，整体active。\n')
 while True:
  st=m.READ(S/'status.json');safe=bool(st.get('running'))
  for item in st.get('running',[]):
   p=S/'runs'/item['job']/'train_log.csv';meta=p.parent/'run_meta.json';logs=list(csv.DictReader(p.open())) if p.exists() else [];safe &= len(logs)<43 and meta.exists() and time.time()-meta.stat().st_mtime>10
  if safe or (S/'independent_audit.json').exists():break
  time.sleep(2)
 m.commit('docs: preserve actual15-model Swin384 merge neural start structure compute and reproducible usage')
 H=m.R/'experiments/ac_merge_contrast_prototype_window01';data={};manifest=[]
 for root in [S,m.M,H]:
  for p in sorted(root.rglob('*')):
   if not p.is_file() or any(x in p.parts for x in ['exports','__pycache__','runs','logs','evaluations']) or p.suffix in ['.pyc','.lock','.tmp'] or p.name in ['continuous_runtime_resources.csv','continuous_runtime_process_io.jsonl','resources.csv','io_resources.jsonl']:continue
   raw=p.read_bytes();name=str(p.relative_to(m.R));data[name]=raw;manifest.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
 out=S/'exports';out.mkdir(exist_ok=True);target=out/'merge_contrast_start_evidence.tar.gz'
 with tarfile.open(target,'w:gz') as tar:
  for name,raw in data.items():
   item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o644;item.uid=item.gid=1000;item.uname=item.gname='login';tar.addfile(item,io.BytesIO(raw))
  raw=json.dumps(dict(files=manifest,start_snapshot=True,not_complete_round=True,prototype_numPy_Haar_and_official_parity_included=True),ensure_ascii=False,indent=2).encode();item=tarfile.TarInfo('EVIDENCE_MANIFEST.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 m.SAVE(out/'export_receipt.json',dict(path=str(target.relative_to(m.R)),sha256=m.sha(target),bytes=target.stat().st_size,files=len(manifest)));print(json.dumps(m.READ(out/'export_receipt.json')),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:m.SAVE(S/'document_start_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
