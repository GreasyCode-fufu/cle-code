from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
S=Path(__file__).resolve().parent;R=S.parents[1];P=R/'experiments/ac_moment_refinement_window01'
sys.path.insert(0,str(R/'experiments/ac_patient_mechanism_evidence_window01'));import patient_evidence as guard;guard.S=S
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def quiet_commit(message):
    while True:
        status=P/'status.json'
        if (P/'bootstrap_failure.json').exists():raise RuntimeError('refinement pipeline failed; inspect it before unrelated CPU work')
        safe=False
        if status.exists():
            st=json.loads(status.read_text());safe=bool(st.get('running'))
            for j in st.get('running',[]):
                import csv
                proc=Path('/proc')/str(j['pid'])/'cmdline';log=P/'runs'/j['job']/'train_log.csv';meta=log.parent/'run_meta.json';rows=list(csv.DictReader(log.open())) if log.exists() else []
                safe &= proc.exists() and j['job'].encode() in proc.read_bytes() and len(rows)<45 and meta.exists() and time.time()-meta.stat().st_mtime>10
            if (P/'independent_audit.json').exists():safe=True
        if safe:break
        time.sleep(5)
    guard.commit(message)
def main():
    assert os.getuid()==1000;save(S/'identity.json',dict(pid=os.getpid(),source_sha256=sha(__file__),started_at=guard.core.timestamp()))
    if not (S/'config.json').exists():
        paths=[S/'crossscale_readout.py',S/'check_crossscale.py',S/'bootstrap_proto.py',S/'PROTO_PROTOCOL_ZH.md',*list((S/'upstream').glob('*'))]
        save(S/'config.json',dict(sources={str(p.relative_to(S)):sha(p) for p in paths if p.is_file()},phase='synthetic CPU prototype only; no formal training efficacy',CLE_images=0,test_prediction_arrays=0,neural_optimizer_steps_on_CLE=0,parent_refinement_config_sha256=sha(P/'config.json')))
    while not (P/'profile/selected.json').exists():
        if (P/'bootstrap_failure.json').exists():raise RuntimeError('parent profile failure; inspect exact log')
        time.sleep(5)
    quiet_commit('research: freeze bounded cross-scale correlation CPU prototype and faithful original operator references')
    if not (S/'numeric_preflight_cpu.json').exists():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        subprocess.run(['/home/login/miniconda3/envs/sunfuxiang/bin/python',str(S/'check_crossscale.py')],cwd=R,env=env,stdin=subprocess.DEVNULL,check=True)
    assert json.loads((S/'numeric_preflight_cpu.json').read_text())['passed']
    quiet_commit('test: verify bounded cross-scale operator NumPy/gradients and exact public moment-model initialization')
    save(S/'complete.json',dict(passed=True,preflight_sha256=sha(S/'numeric_preflight_cpu.json'),formal_neural_training=False,overall_research_complete=False))
if __name__=='__main__':
    try:main()
    except Exception as exc:save(S/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
