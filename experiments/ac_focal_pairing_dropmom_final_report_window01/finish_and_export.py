"""Wait for one own 3-model study, then supplementary metrics and full evidence."""
from pathlib import Path
import fcntl, hashlib, json, os, subprocess, sys, tarfile, time, traceback
F=Path(__file__).resolve().parent;R=F.parents[1];S=R/'experiments/ac_focal_pairing_dropmom_control_window01'
PY='/home/login/miniconda3/envs/sunfuxiang/bin/python'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
    p=F/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def alive(pid,expected):
    p=Path('/proc')/str(pid)/'cmdline'
    return p.exists() and expected.encode() in p.read_bytes()
def call(name,cpu):
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    if cpu:env['CUDA_VISIBLE_DEVICES']=''
    else:env.pop('CUDA_VISIBLE_DEVICES',None)
    subprocess.run([PY,str(F/name)],cwd=R,env=env,stdin=subprocess.DEVNULL,check=True)
def main():
    lock=(F/'finisher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.getuid()==1000 and os.environ['CLE_RESEARCH_OWNER']=='01a171be-ebee-7a13-b05e-97fdb6b75587'
    cfg=read(F/'config.json');assert all(sha(F/p)==v for p,v in cfg['sources'].items())
    save('finisher_identity.json',dict(pid=os.getpid(),source_sha256=sha(__file__),owner=os.environ['CLE_RESEARCH_OWNER'],started_unix=time.time(),training_config_sha256=sha(S/'config.json')))
    while True:
        assert not (S/'bootstrap_failure.json').exists(),'own bootstrap failure; do not restart training'
        complete=all((S/n).exists() and read(S/n).get('passed',True) for n in ['bootstrap_complete.json','independent_audit.json','runtime_sampler_complete_v2.json','usage_smoke/independent_cli_audit.json'])
        identities=[('bootstrap_identity.json','bootstrap_module_structure.py'),('manager_entry_identity_v2.json','run_dropmom_manager.py'),('runtime_sampler_identity_v2.json','continuous_runtime_sampler.py')]
        if complete and all((S/n).exists() and not alive(read(S/n)['pid'],expected) for n,expected in identities):break
        time.sleep(3)
    if not (F/'efficiency_metrics.json').exists():call('benchmark_actual_models.py',False)
    if not (F/'analysis_complete.json').exists():call('analyze_metrics.py',True)
    assert read(F/'analysis_complete.json')['passed'] and read(F/'efficiency_metrics.json')['passed']
    files=set()
    def include(p):
        assert p.is_file() and p.stat().st_uid==1000
        if not any(x in p.parts for x in ['__pycache__','.git','exports']) and p.suffix not in ['.pth','.lock','.tmp','.pyc'] and p.name not in ['finisher.log']:
            files.add(p)
    for root in [S,R/'models/optimize/ac_focal_pairing_dropmom_control_window01',F]:
        for p in root.rglob('*'):
            if p.is_file():include(p)
    studycfg=read(S/'config.json')
    for path in set(studycfg['sources'])|set(studycfg['locked'])|set(read(F/'input_manifest.json')['files'])|set(cfg['locked']):include(R/path)
    manifest={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)}
    save('complete.json',dict(passed=True,training_config_sha256=sha(S/'config.json'),result_sha256=sha(F/'metric_results.json'),efficiency_sha256=sha(F/'efficiency_metrics.json'),models_new=3,epochs_new=180,old_effective_models_retrained=0,overall_research_complete=False))
    include(F/'complete.json');manifest[str((F/'complete.json').relative_to(R))]=dict(sha256=sha(F/'complete.json'),bytes=(F/'complete.json').stat().st_size)
    target=F/'exports';target.mkdir(exist_ok=True)
    archive=target/'focal_dropmom_complete_evidence.tar.gz';assert not archive.exists()
    payload=json.dumps(dict(files=manifest,scope='supplementary true dropMom control and exact screenshot metrics, original packages unchanged'),ensure_ascii=False,indent=2).encode()
    with tarfile.open(archive,'w:gz') as tar:
        import io
        info=tarfile.TarInfo('EVIDENCE_MANIFEST.json');info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
        for p in sorted(files):tar.add(p,arcname=str(p.relative_to(R)),recursive=False)
    save('exports/export_receipt.json',dict(archive=str(archive.relative_to(R)),sha256=sha(archive),bytes=archive.stat().st_size,files=len(files),config_sha256=sha(S/'config.json'),complete_sha256=sha(F/'complete.json'),overall_research_complete=False))
    save('finisher_complete.json',dict(passed=True,receipt_sha256=sha(target/'export_receipt.json'),overall_research_complete=False))

if __name__=='__main__':
    try:main()
    except Exception as e:save('finisher_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
