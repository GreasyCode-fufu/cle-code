from pathlib import Path
import csv,hashlib,io,json,os,sys,tarfile,time
R=Path('/home/login/data/cle/auto/GPT').resolve();C=R/'experiments/ac_crossscale_structure_window01';S=R/'experiments/ac_crossscale_final_report_window01';N=R/'experiments/ac_affine_geometry_structure_window01';S.mkdir(exist_ok=True);sys.path.insert(0,str(C));import run_module_structure as m;m.verify()
assert os.getuid()==1000 and m.READ(C/'independent_audit.json')['passed'] and m.READ(C/'direct_control_finisher_complete.json')['passed']
sys.path.insert(0,str(R/'experiments/ac_texture_error_evidence_window01'));import texture_evidence as guard
def quiet_commit(root,message):
 while True:
  if not (N/'profile/selected.json').exists():break
  st=m.READ(N/'status.json') if (N/'status.json').exists() else {};safe=bool(st.get('running'))
  for item in st.get('running',[]):
   log=N/'runs'/item['job']/'train_log.csv';meta=log.parent/'run_meta.json';rows=list(csv.DictReader(log.open())) if log.exists() else [];safe &= len(rows)<43 and meta.exists() and time.time()-meta.stat().st_mtime>10
  if safe or (N/'independent_audit.json').exists():break
  time.sleep(2)
 guard.S=root;guard.commit(message)
result=m.READ(C/'round_results.json');direct=m.READ(C/'direct_control_comparisons.json');weights=[]
for job in m.READ(C/'jobs.json'):
 assert m.done(job);folder=C/'runs'/job['id'];complete=m.READ(folder/'complete.json');rows=list(csv.DictReader((folder/'train_log.csv').open()));assert len(rows)==60 and complete['best_epoch']==max(range(60),key=lambda k:float(rows[k]['val_macro_f1']))
 assert m.sha(R/complete['checkpoint'])==complete['checkpoint_sha256'];weights.append(dict(job=job['id'],arm=job['arm'],seed=job['seed'],checkpoint=complete['checkpoint'],checkpoint_sha256=complete['checkpoint_sha256'],training_signature=complete['signature'],best_val_epoch=complete['best_epoch'],train_log_sha256=m.sha(folder/'train_log.csv')))
m.SAVE(S/'weights_and_tasks.json',dict(models=12,units=weights,all_actual_weight_sha_verified=True,earliest_val_best_verified=True));budgets={b['arm']:b for b in m.READ(C/'preflight.json')['budgets']}
lines=['# CLE跨尺度统计完整轮：结果、对照与继续依据','','四方法×三配对seed共12模型，全部60epoch和完整427val/576test两路线概率重建及独立计算审计通过。固定B0与原M方向矩有效参考均复用，标签/清单/等级/患者split保持冻结。此轮没有达到目标，原M方向矩保留为工作结构，下一局部几何轮已独立预注册并通过CPU/CUDA预检，整体研究继续。','', '|方法/推理|seed45 F1|seed46|seed47|均值|对固定B0差（百分点）|41患者条件95%区间（百分点）|','|---|---:|---:|---:|---:|---:|---|']
for arm in m.ARMS:
 for mode in ['single','c4']:
  name=arm+'_'+mode;v=result['summary'][name];d=result['comparisons'][name+'_vs_qualification_convnext_tiny_A'];lo,hi=d['conditional_patient_bootstrap']['ci95'];assert not d['numerical_gate'];lines.append('|'+name+'|'+'|'.join(f'{100*x:.5f}%' for x in v['seed_macro_f1']+[v['macro_f1']])+f'|{100*d["mean_delta"]:.5f}|[{100*lo:.5f},{100*hi:.5f}]|')
lines+=['','## 机制与成熟参照','','完整mom_corr的单视图平均94.732707%，比原M方向矩95.873647%低1.140940百分点，三个配对seed均负，患者条件区间[−3.172742,−0.280871]百分点。C4同预算平均94.860102%，相对方向矩96.457925%低1.597823百分点、三个seed均负，区间[−3.848816,−0.488785]。本实现不保留为已确认有效组件；数值初始化/梯度有效并不能替代研究效果。','', 'mom_corr相对去方向矩corr_only单视图约+0.230958百分点，但区间跨0；C4反向约−0.535383百分点且区间跨0，不支持稳定互补。相对等参数打乱空间对应，单视图/C4差分别约−0.348107/−0.765456百分点，区间跨0；未确认真实对应带来预期贡献。相对成熟非居中B-CNN算子，C4三seed均下降约0.944682百分点、条件区间[−2.225164,−0.192147]，中心化/标准化相关在此实现不优于成熟二阶参照。','', '这只否定当前位置/秩/固定尺度/训练配方下的实现与互补假设，不能宣称所有跨尺度/二阶/上下文设计无效。全部控制未超过固定B0严格平均+1.5百分点且稳定区间门槛，不靠最好seed或成熟TTA改写结果。原M方向矩C4的+1.476485百分点仍严格小于1.5，不能舍入称成功。','', '直接累计/去一/打乱/成熟参照在新结果前注册；完成后从逐图概率独立构建整数混淆矩阵，以Fraction重算全部seed差/均值和3/200严格门槛，原始20k共享患者抽样区间同时保存。全部直接比较见DIRECT_CONTROL_REPORT_ZH.md/direct_control_comparisons.json。','', '## 其他主要指标与计算成本','','|方法/推理|accuracy|恶性召回|患者等权accuracy|参数|逻辑GFLOPs|','|---|---:|---:|---:|---:|---:|']
for arm in m.ARMS:
 for mode in ['single','c4']:
  v=result['summary'][arm+'_'+mode];lines.append('|'+arm+'_'+mode+'|'+'|'.join(f'{100*v[k]:.5f}%' for k in ['accuracy','malignant_recall','patient_equal_accuracy'])+f'|{budgets[arm]["parameters"]:,}|{v["counted_flops"]/1e9:.6f}|')
lines+=['','每seed混淆矩阵、完整指标、全部预测见源round_results.json/evaluations。三seed均值不是模型集成，C4每模型四倍前向。逻辑FLOPs按2FLOP/MAC，排除逐元素/归一化/池化/访存，不能当真实耗时。四内核实际测1/2/4并发，总吞吐高并发无有效增益，均选最佳97%的最小并发1；6GiB余量。正式独立连续采样覆盖manager同步推理/患者bootstrap间隙，原profile资源/每epoch等待、计算、吞吐、实际2077索引SHA均保留，启动/收尾的真实低GPU记录不删。','', '## 数据与审计','','固定train2077/182患者、val427/39、test576/41。3595物理缓存只按允许A–C索引访问；全60epoch实际2077唯一索引与SHA检查。原0度val最早最高macroF1选checkpoint，test/C4不重新选epoch。权重有限值、source/config/实际权重SHA、概率两加载路径、患者隔离、预算和login归属独立核验通过。所有12真实weight SHA、任务签名、最佳epoch和训练日志SHA见weights_and_tasks.json，权重二进制仍保留服务器。','', '## 可复现入口和模型使用','','本轮已完成，不恢复旧bootstrap/manager/worker重复训练。原始source/config/注册/CPU/CUDA/并发测量/日志/逐图概率/独立审计完整保留。', '', '```bash','cd /home/login/data/cle/auto/GPT','/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/ac_crossscale_structure_window01/predict_module_model.py --arm mom_corr --seed 45 --image /absolute/path/image.png --views single --output /absolute/path/prediction.json','```','','固定首val单图真实权重CLI已独立核验，对原batch32概率最大差7.15e-7<预设1e-4。脚本只加载完成标记且SHA一致的真实权重，不输出随机未完成模型。权重位置weights/ac_crossscale_structure_window01/<job>/model_best.pth；类别ID0/1/2，恶性ID1。','', '## 研究边界与下一方向','','用户明确test反馈下一结构，历史test已查看且A–C构造用过test标签；所有结果属于自适应内部开发。患者bootstrap仅条件区间，不校正选择偏差，不称独立泛化或全新外部验证。等级只是模型一致性代理，不证明全部医生标签正确。B-CNN/HBP等已具二阶/跨层机制，新实现不能自动称SCI新颖。','', '独立CPU纹理诊断只访问原train/val，train分位阈值不改变等级/清单。高std验证组错误较多但样本/类别混合及多重比较使其只能作为机制线索。局部几何原型已先完成NumPy/独立grid_sample输出/三梯度、double gradcheck、公开Tiny224初始logit及原参数/RNG有限反向；新轮在原M方向矩上检验受约束局部采样，含去方向矩、等参数打乱offset和成熟DCN参照。原型本身没有CLE训练证据；正式配对效果待新轮。不保证提升、论文创新或录用。']
(S/'CROSSSCALE_COMPLETE_REPORT_ZH.md').write_text('\n'.join(lines)+'\n');(S/'.gitignore').write_text('exports/\n__pycache__/\n*.pyc\n')
m.SAVE(S/'final_receipt.json',dict(models=12,source_config_sha256=m.sha(C/'config.json'),full_audit_sha256=m.sha(C/'independent_audit.json'),direct_controls_sha256=m.sha(C/'direct_control_comparisons.json'),all_registered_methods_failed_goal_gate=True,new_goal_threshold_reached=False,weights_manifest_sha256=m.sha(S/'weights_and_tasks.json'),overall_research_complete=False))
quiet_commit(C,'audit: preserve complete crossscale round and direct-control terminal provenance')
quiet_commit(S,'results: retain all12 negative crossscale models and exact cumulative-control evidence')
data={};manifest=[]
for root in [C,R/'models/optimize/ac_crossscale_structure_window01',S]:
 for p in sorted(root.rglob('*')):
  if not p.is_file() or any(x in p.parts for x in ['exports','__pycache__']) or p.suffix in ['.pyc','.lock','.tmp']:continue
  raw=p.read_bytes();name=str(p.relative_to(R));data[name]=raw;manifest.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
out=S/'exports';out.mkdir(exist_ok=True);target=out/'crossscale_complete_evidence.tar.gz'
with tarfile.open(target,'w:gz') as tar:
 for name,raw in data.items():
  item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o644;item.uid=item.gid=1000;item.uname=item.gname='login';tar.addfile(item,io.BytesIO(raw))
 raw=json.dumps(dict(files=manifest,all12_models_complete=True,weights_on_server_by_sha=True),ensure_ascii=False,indent=2).encode();item=tarfile.TarInfo('EVIDENCE_MANIFEST.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
m.SAVE(out/'export_receipt.json',dict(path=str(target.relative_to(R)),sha256=m.sha(target),bytes=target.stat().st_size,files=len(manifest)));print(json.dumps(m.READ(out/'export_receipt.json')),flush=True)
