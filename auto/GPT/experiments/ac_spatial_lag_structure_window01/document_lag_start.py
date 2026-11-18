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
    lines=['# CLE空间间隔关联：结构、实际开训与使用','','此为真实开训快照，不能判断三seed效果。原CNNMom保留，ODD全轮未确认末层奇统计增量；本轮用空间间隔关系作互补假设。',f'快照{receipt["created_at"]}；config SHA {receipt["source_config_sha256"]}。','','```mermaid','flowchart TD',' X[RGB resize256 crop224] --> C[原ConvNeXt加原Mom stem]',' C --> F[末层768x7x7]',' F --> P[原avgpool norm head]',' F --> L[中心化有效重叠对 lag1与3]',' L --> D[两轴归一内积平均及拼接]',' D --> N[无仿射channel LN]',' N --> A[零1536到3投影]',' P --> S[logits相加]',' A --> S','```','','## 冻结结构与训练协议','']
    lines+=(S/'MODULE_PROTOCOL_ZH.md').read_text().splitlines()
    lines+=['','## 正式CPU和CUDA结果','','正式CPU逐字复制原型、真去Mom、原参数对象/公开logit/RNG/有限反向通过；真实CUDA batch32各arm两次有限更新、projection与保留Mom首步梯度正，原Mom与固定B0共六完整427val概率重建严格0；缓存SHA、2077白名单和训练加载拒绝test通过。预检test推理0。','','|arm|参数|逻辑GFLOPs|','|---|---:|---:|']
    for b in cpu['budgets']:lines.append(f'|{b["arm"]}|{b["parameters"]}|{b["counted_flops"]/1e9:.9f}|')
    lines+=['','逻辑2FLOP/MAC含三额外旋转卷积/新Linear，不含重叠统计/LN/逐元素/访存；实际完整目标吞吐另测。','','|arm|1并发图秒|2并发总图秒|4并发总图秒|cap|','|---|---:|---:|---:|---:|']
    for arm in m.ARMS:
        measures=m.READ(S/'profile'/arm/'measurements.json');v={z['slots']:z['aggregate_samples_per_second'] for z in measures};lines.append('|'+arm+'|'+'|'.join(f'{v[k]:.2f}' if k in v else '未继续' for k in [1,2,4])+f'|{policy["slots_by_arm"][arm]}|')
    lines+=['','选择实测最佳97%内最小并发，保留6GiB余量。cap不等于真实并发，每arm仅3seed。实际PID/PPID/launch和GPU/CPU每核/IO/RAM连续采样；此快照'+str(len(recent))+'行GPU平均只是启动前缀，不外推全轮或空队列设计阶段。','','## 使用','','只使用真实完成且SHA核对的权重；管理器已运行，勿重复启动。','','```bash','cd /home/login/data/cle/auto/GPT','/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/ac_spatial_lag_structure_window01/predict_module_model.py --arm mom_lag --seed 45 --image /absolute/path/image.png --views single --output /absolute/path/prediction.json','```','','RGB原图双线性resize256/中心224/ImageNet归一，类别0/1/2、恶性1。C4为四旋转概率均值，非三seed集成。weights/ac_spatial_lag_structure_window01/lag_<arm>_s<seed>/model_best.pth为实际权重路径。固定首val2386真实CLI对原batch32概率另行核验，通过helper后才称使用核验完成。','', '全部test为自适应内部开发，CI不校正选择偏差，非独立泛化或临床验证；没有新颖或录用保证。整体研究active。']
    (S/'STRUCTURE_AND_USE_ZH.md').write_text('\n'.join(lines)+'\n')
    while True:
        st=m.READ(S/'status.json');safe=bool(st.get('running'))
        for i in st.get('running',[]):
            log=S/'runs'/i['job']/'train_log.csv';meta=log.parent/'run_meta.json';rows=list(csv.DictReader(log.open())) if log.exists() else [];safe &= len(rows)<43 and meta.exists() and time.time()-meta.stat().st_mtime>10
        if safe or (S/'independent_audit.json').exists():break
        time.sleep(2)
    m.commit('docs: preserve actual spatial-lag neural start source controls compute recipe and usage')
    data={};manifest=[]
    for root in [S,m.M,m.R/'experiments/ac_spatial_lag_prototype_window01',m.R/'experiments/ac_spatial_lag_final_report_window01']:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or any(v in p.parts for v in ['exports','__pycache__','runs','logs','evaluations']) or p.suffix in ['.pyc','.lock','.tmp'] or p.name in ['continuous_runtime_resources.csv','continuous_runtime_process_io.jsonl','resources.csv','io_resources.jsonl']:continue
            raw=p.read_bytes();name=str(p.relative_to(m.R));data[name]=raw;manifest.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    out=S/'exports';out.mkdir(exist_ok=True);target=out/'spatial_lag_start_evidence.tar.gz'
    with tarfile.open(target,'w:gz') as tar:
        for name,raw in data.items():
            item=tarfile.TarInfo(name);item.size=len(raw);item.uid=item.gid=1000;item.uname=item.gname='login';tar.addfile(item,io.BytesIO(raw))
        raw=json.dumps(dict(files=manifest,start_snapshot=True,not_complete_round=True),ensure_ascii=False,indent=2).encode();item=tarfile.TarInfo('EVIDENCE_MANIFEST.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
    m.SAVE(out/'export_receipt.json',dict(path=str(target.relative_to(m.R)),sha256=m.sha(target),bytes=target.stat().st_size,files=len(manifest)));print(json.dumps(m.READ(out/'export_receipt.json')),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:m.SAVE(S/'document_start_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
