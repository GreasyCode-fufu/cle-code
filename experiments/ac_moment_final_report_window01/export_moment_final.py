from pathlib import Path
import csv,hashlib,io,json,os,sys,tarfile,time
R=Path('/home/login/data/cle/auto/GPT').resolve();P=R/'experiments/ac_moment_refinement_window01';S=R/'experiments/ac_moment_final_report_window01';N=R/'experiments/ac_crossscale_structure_window01'
assert os.getuid()==1000;S.mkdir(exist_ok=True);sys.path.insert(0,str(P));import run_module_structure as m;m.verify();assert m.READ(P/'independent_audit.json')['passed']
sys.path.insert(0,str(R/'experiments/ac_patient_mechanism_evidence_window01'));import patient_evidence as guard
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quiet_commit(root,message):
    while True:
        st=m.READ(N/'status.json') if (N/'status.json').exists() else None
        if st is None:
            if not (N/'profile/selected.json').exists():break
        elif st.get('running'):
            safe=True
            for j in st['running']:
                log=N/'runs'/j['job']/'train_log.csv';meta=log.parent/'run_meta.json';n=len(list(csv.DictReader(log.open()))) if log.exists() else 0
                safe &= n<45 and meta.exists() and time.time()-meta.stat().st_mtime>10
            if safe:break
        elif (N/'independent_audit.json').exists():break
        time.sleep(3)
    guard.S=root;guard.commit(message)
result=m.READ(P/'round_results.json');mechanism=m.READ(P/'mechanism_comparisons.json');weights=[]
for job in m.READ(P/'jobs.json'):
    assert m.done(job);folder=P/'runs'/job['id'];c=m.READ(folder/'complete.json');rows=list(csv.DictReader((folder/'train_log.csv').open()));assert len(rows)==60 and c['best_epoch']==max(range(60),key=lambda i:float(rows[i]['val_macro_f1']))
    assert sha(R/c['checkpoint'])==c['checkpoint_sha256'];weights.append(dict(job=job['id'],arm=job['arm'],seed=job['seed'],checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256'],training_signature=c['signature'],best_val_epoch=c['best_epoch'],train_log_sha256=sha(folder/'train_log.csv')))
m.SAVE(S/'weights_and_tasks.json',dict(units=weights,models=12,all_actual_weight_sha_verified=True,all_task_signatures_verified=True))
lines=['# CLE方向矩细化轮：完整结果与继续研究依据','','四方法×三配对seed共12模型，全部60epoch完成并独立审计通过。冻结B0和原M方向矩等参考均复用，不因阴性改变baseline或A–C清单。整体研究继续，下一轮跨尺度对应统计已冻结并进入数值预检/吞吐实测流水线。','', '|方法/推理|seed45宏F1|seed46|seed47|均值|相对B0差（百分点）|41患者条件95%区间（百分点）|','|---|---:|---:|---:|---:|---:|---|']
for arm in m.ARMS:
    for mode in ['single','c4']:
        name=arm+'_'+mode;r=result['summary'][name];c=result['comparisons'][name+'_vs_qualification_convnext_tiny_A'];lo,hi=c['conditional_patient_bootstrap']['ci95'];lines.append('|'+name+'|'+'|'.join(f'{100*v:.5f}%' for v in r['seed_macro_f1']+[r['macro_f1']])+f'|{100*c["mean_delta"]:.5f}|[{100*lo:.5f},{100*hi:.5f}]|')
lines+=['','## 机制判断','','原方向矩工作结构保留。mean_only删除方向标准差、用重复均值替代，在同18528新增参数下，单视图对原方向矩平均−0.839786百分点，三seed均负、条件区间[−2.010478,−0.209912]百分点，支持标准差具有增量。','', 'random_moments用固定随机空间核排列代替真实C4方向关系。单视图与原方向矩差区间跨0，不能宣称真实旋转关系已普遍确认；C4同预算推理时三seed均下降，平均−0.522686百分点、条件区间[−1.033337,−0.148824]，属于有限支持。','', '叠加ECA单视图平均+0.114535百分点，相对方向矩仅seed45正、区间跨0；C4三seed都下降，平均−0.532605百分点。额外旋转增强未显示稳定增益，单视图−0.353871、C4−0.462958百分点，相对原方向矩区间均跨0。本轮不将ECA/更强旋转增强保留为已确认有效增量，全部阴性保留。','', '本轮没有新结构达到相对固定B0严格平均+1.5百分点及稳定条件门槛。原M方向矩+C4为+1.476485百分点，不能舍入为达标。下一轮保留原方向矩，检验跨stage3/4空间对应统计，并以去方向矩/固定打乱对应/成熟双线性算子重训对照验证。','', '## 指标、预算和审计','','|方法/推理|accuracy|恶性召回|患者等权accuracy|参数|逻辑GFLOPs|','|---|---:|---:|---:|---:|---:|']
budgets={b['arm']:b for b in m.READ(P/'preflight.json')['budgets']}
for arm in m.ARMS:
    for mode in ['single','c4']:
        r=result['summary'][arm+'_'+mode];lines.append('|'+arm+'_'+mode+'|'+'|'.join(f'{100*r[k]:.5f}%' for k in ['accuracy','malignant_recall','patient_equal_accuracy'])+f'|{budgets[arm]["parameters"]:,}|{r["counted_flops"]/1e9:.6f}|')
lines+=['','逐seed混淆矩阵/指标见源round_results.json，直接机制患者配对比较见mechanism_comparisons.json/MECHANISM_REPORT_ZH.md。三seed均值不是多模型集成；C4是每个checkpoint四倍前向。FLOPs为2FLOP/MAC，排除逐元素/归一化/池化/访存，不等于实际运行耗时，性能实测全部保留。','', '固定train2077/182患者、val427/39、test576/41，split患者互斥。每epoch核验2077唯一允许索引及SHA，3595物理缓存不等于训练使用3595。原0度val最早最高macroF1选权重，无test选epoch；完整427val/576test逐图概率两加载/变换路线重建，有限权重/sourceSHA/checkpointSHA/login归属全部通过。','', 'CPU/CUDA真实AMP/六个完整baseline与工作结构概率复现均先通过。四类内核各测1/2/4并发并选最佳97%最少并发，真实资源记录保留。独立额外采样启动故障v1已保留，v2仅修startup等待status，于01:26至01:41覆盖446条；前段依赖manager/每epoch原资源记录，不能称额外collector全程覆盖。','', '## 可复现命令与使用','','权重仍在服务器weights/ac_moment_refinement_window01/<job>/model_best.pth，全部12文件真实SHA/签名/最早val最佳epoch见weights_and_tasks.json。本轮已完成，不要恢复旧manager/worker来重复训练。源码/config/失败记录都保留。','', '```bash','cd /home/login/data/cle/auto/GPT','/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/ac_moment_refinement_window01/predict_module_model.py --arm moments_eca --seed 45 --image /absolute/path/image.png --views single --output /absolute/path/prediction.json','```','','预测脚本加载完成checkpoint并检查SHA，基础脚本原M轮真实CLI已核验；P轮本拷贝尚未另做单图CLI smoke，不将它说成已独立完成使用验证。全部逐图正式概率复算仍已经独立审计。','', '## 解释边界','','用户明确test参与下一设计；历史test已查看且A–C清单构造也使用其标签。所有bootstrap仅条件描述，未校正适应性筛选偏差，不能称独立泛化或全新外部临床验证。等级是模型一致性代理，不证明医生标签全部正确。方向矩/二阶统计存在成熟背景，不能将自主实现或组合直接称已证明新颖。不保证SCI创新或录用，整体目标保持进行。']
(S/'MOMENT_COMPLETE_REPORT_ZH.md').write_text('\n'.join(lines)+'\n');(S/'.gitignore').write_text('exports/\n__pycache__/\n*.pyc\n');m.SAVE(S/'final_receipt.json',dict(models=12,source_config_sha256=sha(P/'config.json'),source_audit_sha256=sha(P/'independent_audit.json'),results_sha256=sha(P/'round_results.json'),mechanism_comparison_sha256=sha(P/'mechanism_comparisons.json'),weight_manifest_sha256=sha(S/'weights_and_tasks.json'),new_goal_threshold_reached=False,overall_research_complete=False))
quiet_commit(P,'results: retain completed12-model moment refinements and runtime repair history')
quiet_commit(S,'results: export full paired moment refinements and honest negative cumulative evidence')
data={};files=[]
for root in [P,R/'models/optimize/ac_moment_refinement_window01',S]:
    for p in sorted(root.rglob('*')):
        if not p.is_file() or any(v in p.parts for v in ['__pycache__','exports']) or p.suffix in ['.pyc','.lock','.tmp']:continue
        raw=p.read_bytes();name=str(p.relative_to(R));data[name]=raw;files.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
out=S/'exports';out.mkdir(exist_ok=True);target=out/'moment_complete_evidence.tar.gz'
with tarfile.open(target,'w:gz') as tar:
    for name,raw in data.items():
        info=tarfile.TarInfo(name);info.size=len(raw);info.uid=info.gid=1000;info.uname=info.gname='login';info.mode=0o644;tar.addfile(info,io.BytesIO(raw))
    raw=json.dumps(dict(files=files,all12_models_completed=True,weight_binaries_stored_on_server_by_sha=True),ensure_ascii=False,indent=2).encode();info=tarfile.TarInfo('EVIDENCE_MANIFEST.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
m.SAVE(out/'export_receipt.json',dict(path=str(target.relative_to(R)),sha256=sha(target),bytes=target.stat().st_size,files=len(files)));print(json.dumps(m.READ(out/'export_receipt.json')),flush=True)
