from pathlib import Path
import csv,hashlib,io,json,os,sys,tarfile,time
import numpy as np
R=Path('/home/login/data/cle/auto/GPT').resolve();M=R/'experiments/ac_module_structure_window01';P=R/'experiments/ac_moment_refinement_window01';S=R/'experiments/ac_module_final_report_window01'
assert os.getuid()==1000;S.mkdir(exist_ok=True);sys.path.insert(0,str(M));import run_module_structure as m;m.verify();assert m.READ(M/'independent_audit.json')['passed']
sys.path.insert(0,str(R/'experiments/ac_patient_mechanism_evidence_window01'));import patient_evidence as guard
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe_commit(root,message):
    while True:
        st=m.READ(P/'status.json');safe=bool(st.get('running'))
        for j in st.get('running',[]):
            q=Path('/proc')/str(j['pid'])/'cmdline';log=P/'runs'/j['job']/'train_log.csv';meta=log.parent/'run_meta.json';rows=list(csv.DictReader(log.open())) if log.exists() else []
            safe &= q.exists() and j['job'].encode() in q.read_bytes() and len(rows)<45 and meta.exists() and time.time()-meta.stat().st_mtime>10
        if safe or (P/'independent_audit.json').exists():break
        assert not st.get('failures');time.sleep(5)
    guard.S=root;guard.commit(message)

result=m.READ(M/'round_results.json');cfg=m.READ(M/'config.json');weights=[]
for job in m.READ(M/'jobs.json'):
    assert m.done(job);folder=M/'runs'/job['id'];c=m.READ(folder/'complete.json');log=list(csv.DictReader((folder/'train_log.csv').open()));assert len(log)==60 and c['best_epoch']==max(range(60),key=lambda i:float(log[i]['val_macro_f1']))
    assert sha(R/c['checkpoint'])==c['checkpoint_sha256'];weights.append(dict(job=job['id'],arm=job['arm'],seed=job['seed'],checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256'],training_signature=c['signature'],best_val_epoch=c['best_epoch'],train_log_sha256=sha(folder/'train_log.csv')))
m.SAVE(S/'weights_and_tasks.json',dict(units=weights,models=18,all_frozen_signatures_verified=True,all_actual_weight_sha_verified=True))
lines=['# CLE共享模块与方向矩：完整18模型研究报告','','本轮全部6方法×3seed已完成60epoch并通过独立计算审计。固定B0保持ConvNeXt-Tiny224原配方，旧有效结果复用，不重训或降低baseline。整个持续研究尚未完成，后续方向矩细化轮已经实际训练。','', '|方法/推理|seed45宏F1|seed46|seed47|均值|相对B0差（百分点）|41患者条件95%区间（百分点）|','|---|---:|---:|---:|---:|---:|---|']
for arm in m.ARMS:
    for mode in ['single','c4']:
        name=arm+'_'+mode;r=result['summary'][name];c=result['comparisons'][name+'_vs_qualification_convnext_tiny_A'];lo,hi=c['conditional_patient_bootstrap']['ci95']
        lines.append('|'+name+'|'+'|'.join(f'{100*v:.4f}%' for v in r['seed_macro_f1']+[r['macro_f1']])+f'|{100*c["mean_delta"]:.4f}|[{100*lo:.4f}, {100*hi:.4f}]|')
lines+=['','## 当前证据及后续动作','','方向矩单视图平均95.873647%，对固定B0+0.892207百分点，三seed方向正，条件区间[+0.129863,+2.397088]百分点。它有继续验证的信号，但对普通随机旋转训练仅+0.189753百分点，区间跨0，不能据此称已证明独立创新。','', '方向矩四视图平均96.457925%，对固定B0+1.476485百分点，严格低于1.5。不得舍入、挑seed或放宽标准宣称达标。相同C4预算下对B0 C4平均+0.531505百分点，区间跨0，也必须披露。成熟Swin C4的此前条件数值门槛另列，不能替代本轮结构有效性。','', 'ECA三个seed单视图方向一致，平均+0.468472百分点，但区间跨0；保留为探索性参考，不称确认有效。WTConv/SimAM/EMA的配对不确定性也完整保存，不以最好seed挑选。','', '新细化轮ac_moment_refinement_window01保留方向矩工作结构，重训mean_only/random_moments两个等参数消融，并训练moments_eca、aug_moments互补对照，共12项；原B0、moments/ECA/旋转增强的有效三seed全部复用。代码、预注册、CPU与真实CUDA预检、全部原B0和工作moments六个完整427概率重建已先通过，再正式开训。','', '## 其它主要指标','','|方法/推理|accuracy|恶性召回|患者等权accuracy|参数|逻辑GFLOPs|','|---|---:|---:|---:|---:|---:|']
budgets={b['arm']:b for b in m.READ(M/'preflight.json')['budgets']}
for arm in m.ARMS:
    for mode in ['single','c4']:
        r=result['summary'][arm+'_'+mode];lines.append('|'+arm+'_'+mode+'|'+'|'.join(f'{100*r[k]:.4f}%' for k in ['accuracy','malignant_recall','patient_equal_accuracy'])+f'|{budgets[arm]["parameters"]:,}|{r["counted_flops"]/1e9:.6f}|')
lines+=['','混淆矩阵、每seed指标、对同C4的B0/DINO/Swin强参照比较、全部患者配对差和逐图预测见源round_results.json及evaluations。所有三seed均值不是三模型集成，C4为每模型4倍前向。FLOPs按2FLOP/MAC，明确排除norm/pooling/activation/逐元素/softmax/访存；实际代价同时看性能profile和日志。','', '## 数据、资源与审计','','固定train2077/182患者、val427/39、test576/41，全程保持原A–C患者隔离、标签、等级/阈值和清单。每正式epoch记录2077个唯一允许索引及SHA，物理3595缓存的整体校验不是3595训练样本。最佳checkpoint按原0度val最早最高macroF1，模型有限，完整val/test概率沿两个独立变换/加载路径重建，source/权重SHA、预算和login归属核验通过。','', '各类实际测1/2/4并发，总吞吐高并发无有效增益，均按最佳97%最少并发选择1，显存余6GiB。manager资源与每epoch加载等待/计算/吞吐持续记录；独立额外采样仅覆盖2026-10-25 00:31至00:55的704个样本，之前依赖manager/per-epoch记录，不能虚称额外采样全程无缺口。','', '## 复现与模型使用','','全部18模型已完成，不要重新启动本轮训练来“恢复”。权重在服务器weights/ac_module_structure_window01/<job>/model_best.pth；weights_and_tasks.json给出全部真实文件SHA、任务签名和最早val最佳epoch。模型代码与配置保持冻结，原CPU断言和CSV接口修订的v1原件/日志均保留；实际审计使用postprocessing_revision_v2声明的版本。','', '```bash','cd /home/login/data/cle/auto/GPT','/home/login/miniconda3/envs/sunfuxiang/bin/python \\','  experiments/ac_module_structure_window01/predict_module_model.py \\','  --arm c4_moments --seed 45 --image /absolute/path/image.png \\','  --views single --output /absolute/path/prediction.json','```','','源脚本WTConv首个真实checkpoint的固定首张val单图使用已独立核验；所有模型加载严格检查完成标记与SHA，未完成不会返回随机权重预测。当前运行的细化轮入口是ac_moment_refinement_window01/bootstrap_module_structure.py，先核对实际/proc身份，不重复启动。','', '## 解释边界','','用户明确要求test参与后续结构开发；历史test已查看，A–C构建也使用过test标签。等级仅模型一致性代理，不证明医生标签全正确。所有患者bootstrap区间是适应性选择后的条件描述，不能称全新外部/独立泛化确认。独立审计指计算复核，不是新临床验证。','', '既有群lifting、ECA、小波、SimAM、EMA和二阶统计均有成熟背景；自主实现或拼接不自动证明新颖。方向矩还需等参数及成熟增强对照，新的跨尺度相关读出目前仅有CPU数值原型，尚无CLE训练结果。继续机制研究，不保证数值目标或SCI创新、录用。']
(S/'MODULE_COMPLETE_REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(S/'.gitignore').write_text('exports/\n__pycache__/\n*.pyc\n')
m.SAVE(S/'final_receipt.json',dict(models=18,source_config_sha256=sha(M/'config.json'),source_audit_sha256=sha(M/'independent_audit.json'),source_results_sha256=sha(M/'round_results.json'),weight_manifest_sha256=sha(S/'weights_and_tasks.json'),best_new_component_singleview_mean_delta=.008922072902163625,best_new_component_c4_mean_delta=.01476484750113923,reliable_goal_threshold_reached=False,overall_research_complete=False))
safe_commit(S,'results: retain complete18-model module study, strict near-threshold failure and true weight provenance')
data={};files=[]
for root in [M,R/'models/optimize/ac_module_window01',S]:
    for q in sorted(root.rglob('*')):
        if not q.is_file() or any(x in q.parts for x in ['__pycache__','exports']) or q.suffix in ['.pyc','.lock','.tmp']:continue
        name=str(q.relative_to(R));raw=q.read_bytes();data[name]=raw;files.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
out=S/'exports';out.mkdir(exist_ok=True);target=out/'module_complete_evidence.tar.gz'
with tarfile.open(target,'w:gz') as tar:
    for name,raw in data.items():
        info=tarfile.TarInfo(name);info.size=len(raw);info.uid=info.gid=1000;info.uname=info.gname='login';info.mode=0o644;tar.addfile(info,io.BytesIO(raw))
    raw=json.dumps(dict(files=files,all18_models_completed=True,weight_binaries_stored_on_server_by_sha=True),ensure_ascii=False,indent=2).encode();info=tarfile.TarInfo('EVIDENCE_MANIFEST.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
m.SAVE(out/'export_receipt.json',dict(path=str(target.relative_to(R)),sha256=sha(target),bytes=target.stat().st_size,files=len(files)));print(json.dumps(m.READ(out/'export_receipt.json')),flush=True)
