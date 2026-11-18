"""Actual neural-start snapshot only. Does not evaluate effectiveness."""
from pathlib import Path
import csv,io,json,os,sys,tarfile,time,traceback,hashlib
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
def main():
    m.verify();assert os.getuid()==1000
    while True:
        assert not (S/'bootstrap_failure.json').exists()
        if (S/'status.json').exists() and (S/'continuous_runtime_resources.csv').exists():
            st=m.READ(S/'status.json');items=st.get('running',[])
            ready=bool(items) and all((S/'runs'/i['job']/'train_log.csv').exists() for i in items)
            if ready and list(csv.DictReader((S/'continuous_runtime_resources.csv').open())):break
        time.sleep(2)
    workers=[]
    for i in items:
        p=Path('/proc')/str(i['pid']);assert p.stat().st_uid==1000 and i['job'].encode() in (p/'cmdline').read_bytes()
        rows=list(csv.DictReader((S/'runs'/i['job']/'train_log.csv').open()));assert rows and all(int(v['actual_train_count'])==int(v['unique_train_count'])==2077 for v in rows)
        workers.append(dict(job=i['job'],pid=i['pid'],epochs=len(rows),index_sha256=rows[-1]['train_index_set_sha256']))
    pre=m.READ(S/'preflight.json');cpu=m.READ(S/'module_source_audit.json');assert pre['passed'] and cpu['passed']
    policy=m.READ(S/'profile/selected.json');recent=list(csv.DictReader((S/'continuous_runtime_resources.csv').open()))[-60:]
    receipt=dict(created_at=m.core.timestamp(),source_config_sha256=m.sha(S/'config.json'),CPU_audit_sha256=m.sha(S/'module_source_audit.json'),CUDA_audit_sha256=m.sha(S/'preflight.json'),models=15,verified_live_workers=workers,profile_policy=policy,startup_resource_rows=len(recent),startup_mean_gpu_util=sum(float(v['gpu_util']) for v in recent)/len(recent),real_training_started=True,three_seed_effectiveness_not_established=True,overall_research_complete=False)
    m.SAVE(S/'NEURAL_START_RECEIPT.json',receipt)
    lines=['# CLE末层空间奇统计：结构、实际开训与使用','','MERGE全轮完成、同主干视图匹配区间未确认新组件；保留原CNN方向矩工作机制，新增一个末层空间分布描述。五arm×45/46/47共15模型，所有seed和阴性保留。启动凭据只能证明实训已开始，不证明效果。','',f'快照：{receipt["created_at"]}；config SHA {receipt["source_config_sha256"]}。','', '## 数据流','','```mermaid','flowchart TD',' X[RGB resize256 随机或中心224] --> C[原ConvNeXt加原Mom stem]',' C --> Z[末层768x7x7特征]',' Z --> P[原avgpool learned norm head]',' Z --> T[逐通道空间标准化z]',' T --> D[mean tanh z 的三次方]',' D --> N[无仿射channel LN]',' N --> A[零768到3投影]',' P --> S[logits相加]',' A --> S','```','','z=(x−空间μ)/sqrt(空间v+1e−6)，d=mean(tanh(z)^3)，再无仿射channel LN eps1e−5。原avgpool/learned norm/head对象完整保留。mom_odd累计；odd_only真去18528方向矩；mom_raw3相同z三阶矩；mom_std相同空间标准差减.001；mom_even相同tanh偶二阶去符号。四统计同LN/同2307参数零投影，原公开logit/RNG/原参数ID一致，未热启动CLE权重。','', '|arm|参数|逻辑GFLOPs|','|---|---:|---:|']
    for b in cpu['budgets']:lines.append(f'|{b["arm"]}|{b["parameters"]}|{b["counted_flops"]/1e9:.9f}|')
    lines+=['','逻辑2FLOP/MAC含三额外旋转卷积和新Linear，不含统计/tanh/LN/逐元素/访存；实际吞吐另测。原Mom/B0的有效三seed复用，固定B0预算与选择规则不改；成熟C4参照同时保留。','', '## 训练、审计和边界','','固定AC train2077/182患者、val427/39、test576/41，恶性1、患者可多标签。原标签/等级/阈值/清单/split/采样不变。物理3595缓存不是训练3595；每epoch2077唯一索引SHA e8fe76bbcf6d8faf8dcb6f38a148341ffe81d10ac0cdc3e4dc38e9f8ccafd691。原Q/M A224公开ImageNet，stochastic depth .1，60epoch/batch32，encoder1e-5、新模块/head1e-4，AdamW.05/cosine0/clip1/AMP、HV翻转/rotation15/randomcrop224、workers2/thread1。','', '原0度val最早最高macroF1选权重；三seed全部完成后注册实际checkpoint，再读完整576test四角度，独立直接像素与冻结T TestData A两路线重建。直接控制mom_odd对原Mom/odd_only/raw3/std/even/B0两mode，整数Fraction严格>3/200、三seed方向及41患者20k条件CI，强同预算参照不能省略。test按用户授权反馈下一设计，历史查看且AC构建用标签；全部自适应内部开发，CI不校正选择偏差，不是独立泛化或外部验证。','', '正式CPU七变体公开起点/有限反向/真去Mom/参数预算通过；原型NumPy、反例、gradcheck、退化有限通过。CUDA真实batch32两次有限更新及新投影/Mom梯度非零、原Mom与B0六完整427val概率误差均0、训练加载拒绝test、缓存SHA通过。不是仅凭结构名推断兼容。','', '[Spatial Moment Pooling作者论文](https://arxiv.org/abs/2209.14583)已有高阶空间矩和LN。raw3只是受控接口适配，不是作者多阶拼接/IQA完整复现，也未确认作者代码。tanh、矩、LN、零残差及组合不自动新颖。d在LN前[-1,1]、LN后无此界；非Pearson偏度、不保证稳健影响函数、病灶定位或整网等变。','', '## 真实吞吐选择','','|arm|1并发图秒|2并发总图秒|4并发总图秒|cap|','|---|---:|---:|---:|---:|']
    for arm in m.ARMS:
        measures=m.READ(S/'profile'/arm/'measurements.json');v={z['slots']:z['aggregate_samples_per_second'] for z in measures};lines.append('|'+arm+'|'+'|'.join(f'{v[k]:.2f}' if k in v else '未继续' for k in [1,2,4])+f'|{policy["slots_by_arm"][arm]}|')
    lines+=['', '选择实测最佳97%内最小并发，保留6GiB余量。每arm仅三seed，cap不等于真实并发；资源连续保留实际PID/PPID/launch、GPU/CPU每核/IO/RAM。此处启动'+str(len(recent))+'行平均GPU仅短快照，不外推全轮，最终报告按时间加权保留低GPU区间。','', '## 使用','','管理器已运行，勿重复启动。仅使用真实已完成、SHA核对的权重：','','```bash','cd /home/login/data/cle/auto/GPT','/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/ac_spatial_odd_moment_structure_window01/predict_module_model.py --arm mom_odd --seed 45 --image /absolute/path/image.png --views single --output /absolute/path/prediction.json','```','','RGB原图resize256双线性/中心224/ImageNet归一，类别0/1/2、恶性1；C4四旋转概率均值，非三seed集成。首固定val2386真实checkpoint CLI将单独比对原batch32概率，helper通过后才称使用核验完成。权重位于weights/ac_spatial_odd_moment_structure_window01/odd_<arm>_s<seed>/model_best.pth。整体active，不保证提点、SCI新颖或录用。']
    (S/'STRUCTURE_AND_USE_ZH.md').write_text('\n'.join(lines)+'\n')
    while True:
        st=m.READ(S/'status.json');safe=bool(st.get('running'))
        for i in st.get('running',[]):
            log=S/'runs'/i['job']/'train_log.csv';meta=log.parent/'run_meta.json';rows=list(csv.DictReader(log.open())) if log.exists() else [];safe &= len(rows)<43 and meta.exists() and time.time()-meta.stat().st_mtime>10
        if safe or (S/'independent_audit.json').exists():break
        time.sleep(2)
    m.commit('docs: preserve actual spatial-odd neural start source controls compute recipe and usage')
    data={};manifest=[]
    for root in [S,m.M,m.R/'experiments/ac_spatial_odd_moment_prototype_window01']:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or any(v in p.parts for v in ['exports','__pycache__','runs','logs','evaluations']) or p.suffix in ['.pyc','.lock','.tmp'] or p.name in ['continuous_runtime_resources.csv','continuous_runtime_process_io.jsonl','resources.csv','io_resources.jsonl']:continue
            raw=p.read_bytes();name=str(p.relative_to(m.R));data[name]=raw;manifest.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    out=S/'exports';out.mkdir(exist_ok=True);target=out/'spatial_odd_start_evidence.tar.gz'
    with tarfile.open(target,'w:gz') as tar:
        for name,raw in data.items():
            item=tarfile.TarInfo(name);item.size=len(raw);item.uid=item.gid=1000;item.uname=item.gname='login';tar.addfile(item,io.BytesIO(raw))
        raw=json.dumps(dict(files=manifest,start_snapshot=True,not_complete_round=True),ensure_ascii=False,indent=2).encode();item=tarfile.TarInfo('EVIDENCE_MANIFEST.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
    m.SAVE(out/'export_receipt.json',dict(path=str(target.relative_to(m.R)),sha256=m.sha(target),bytes=target.stat().st_size,files=len(manifest)));print(json.dumps(m.READ(out/'export_receipt.json')),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:m.SAVE(S/'document_start_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
