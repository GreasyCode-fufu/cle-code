"""Export a byte-consistent own-study evidence snapshot; does not include trained weights."""
from pathlib import Path
import hashlib,io,json,os,sys,tarfile,time
S=Path(__file__).resolve().parent;R=S.parents[1];sys.path.insert(0,str(S))
import run_module_structure as m
def main():
    cfg=m.verify();cpu=m.READ(S/'module_source_audit.json');gpu=m.READ(S/'preflight.json');assert cpu['passed'] and gpu['passed']
    for row in m.READ(S/'shared_snapshot_manifest.json'):assert m.sha(Path(row['source']))==row['source_sha256']==m.sha(R/row['copy'])
    previous=m.READ(m.F/'queue_status_v2.json');current=m.READ(S/'bootstrap_status_v2.json')
    identity=m.READ(S/'bootstrap_identity_v2.json');proc=Path('/proc')/str(identity['pid'])/'cmdline';assert proc.exists() and b'bootstrap_module_structure_v2.py' in proc.read_bytes()
    receipt=dict(timestamp=m.core.timestamp(),config_sha256=m.sha(S/'config.json'),jobs_sha256=m.sha(S/'jobs.json'),cpu_audit_sha256=m.sha(S/'module_source_audit.json'),gpu_preflight_sha256=m.sha(S/'preflight.json'),coordinator_identity=identity,coordinator_live=True,status=current,previous_queue=previous,new_formal_trainings_completed=len(list((S/'runs').glob('*/complete.json'))) if (S/'runs').exists() else 0,overall_research_complete=False)
    m.SAVE(S/'START_RECEIPT.json',receipt)
    text='''# 共享模块正式研究账本（启动快照）

本轮在冻结ConvNeXt-Tiny B0基础上注册WTConv、SimAM、ECA、EMA四类机制及方向矩/成熟方向增强对照，共18项60epoch训练。模型、数据和配方已冻结并提交。

- da9115c：共享/官方原件、差异审计、模型源码、预注册、任务和配置。
- a34c00d：保留CPU v1 ECA逐位断言失败及4.44e-16舍入诊断，新增checker/bootstrap v2；网络与食谱未改。
- 34b9970：全部CPU算子、梯度、Haar独立参照、预算、基础图像与随机数状态检查通过。
- 3e44ed5：真实batch32 CUDA所有arm两步AMP有限更新；全部3seed原B0/427val概率重建；缓存SHA及test训练访问拒绝。

当前新模块正式60epoch训练尚未开始：bootstrap_v2已存活并排队，在原几何轮有效模型自然完成及审计通过后，隔离实测每类并发吞吐并自动开训。不能把已完成的CUDA数值更新当成正式训练结果。实时状态以START_RECEIPT及服务器/proc、bootstrap_status_v2、后续status文件为准。

共享库只读，复制前后SHA相同。新增模型及运行目录独立。所有训练仅用固定2077张AC train；test按用户要求作每方法开发反馈，非独立泛化确认。库模块参照不自动等于SCI创新；有稳定效果后保留，再设计互补机制和重训消融。整体任务未完成，持续研究未停止。

结构、预算、复现和完成后权重用法见MODULE_STRUCTURE_AND_USE_ZH.md；全部失败及修正原件保留，不盲重训有效模型。当前证据包不包含尚未产生的新模型权重。
'''
    (S/'RESEARCH_LEDGER.md').write_text(text,encoding='utf-8')
    (S/'.gitignore').write_text('__pycache__/\n*.pyc\nexports/\n*.lock\n',encoding='utf-8')
    m.commit('docs: retain module study ledger, reproducible model use and verified live continuation receipt')
    files=[];data={}
    for root in [S,m.M]:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or any(n in ['__pycache__','exports'] for n in p.parts) or p.suffix in ['.pyc','.lock','.tmp']:continue
            name=str(p.relative_to(R));raw=p.read_bytes();data[name]=raw;files.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),uid=p.stat().st_uid))
    manifest=dict(created_at=m.core.timestamp(),files=files,checkpoint_binaries_included=False,baseline_weights_referenced_by_sha=True,new_models_not_trained_yet=True)
    folder=S/'exports';folder.mkdir(exist_ok=True);part=folder/'module_start_evidence.partial.tar.gz';target=folder/'module_start_evidence.tar.gz'
    with tarfile.open(part,'w:gz') as tar:
        for name,raw in data.items():
            info=tarfile.TarInfo(name);info.size=len(raw);info.uid=info.gid=1000;info.uname=info.gname='login';info.mode=0o644;info.mtime=int(time.time());tar.addfile(info,io.BytesIO(raw))
        raw=json.dumps(manifest,ensure_ascii=False,indent=2).encode();info=tarfile.TarInfo('EVIDENCE_MANIFEST.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    os.replace(part,target);m.SAVE(folder/'export_receipt.json',dict(sha256=m.sha(target),bytes=target.stat().st_size,files=len(files),path=str(target.relative_to(R)),snapshot_receipt=receipt))
    print(json.dumps(m.READ(folder/'export_receipt.json'),ensure_ascii=False))
if __name__=='__main__':main()
