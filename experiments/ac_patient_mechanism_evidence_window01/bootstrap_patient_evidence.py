"""Own-window CPU evidence pipeline; no CUDA visibility and no model optimization."""
from pathlib import Path
import fcntl,json,os,subprocess,time
S=Path(__file__).resolve().parent;PY='/home/login/miniconda3/envs/sunfuxiang/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
def save(v):
    v.update(pid=os.getpid(),updated_at_unix=time.time());p=S/'bootstrap_status.json';part=p.with_suffix('.tmp');part.write_text(json.dumps(v,indent=2));os.replace(part,p)
with (S/'bootstrap.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,stage in [('patient_evidence.py','prepare'),('patient_evidence.py','preflight'),('patient_evidence.py','extract'),('analyze_patient_evidence.py',None)]:
        save(dict(phase=stage or 'analysis'));cmd=[PY,'-u',str(S/name)]+([stage] if stage else [])
        code=subprocess.call(cmd,cwd=S.parents[1],env=env)
        if code:save(dict(phase='fault',failed_stage=stage or 'analysis',returncode=code));raise SystemExit(code)
    save(dict(phase='evidence_completed_continue_network_research',network_updates=0,test_images=0,overall_research_complete=False))
