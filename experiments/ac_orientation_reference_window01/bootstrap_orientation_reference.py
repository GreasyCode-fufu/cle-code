from pathlib import Path
import fcntl,json,os,subprocess,time
S=Path(__file__).resolve().parent;PY='/home/login/miniconda3/envs/sunfuxiang/bin/python'
with (S/'bootstrap.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for stage in ['prepare','preflight','evaluate','analyze']:
        (S/'bootstrap_status.json').write_text(json.dumps(dict(phase=stage,pid=os.getpid(),updated_at=time.time()),indent=2))
        code=subprocess.call([PY,'-u',str(S/'run_orientation_reference.py'),stage],cwd=S.parents[1],env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        if code:
            (S/'bootstrap_status.json').write_text(json.dumps(dict(phase='fault',stage=stage,returncode=code,pid=os.getpid()),indent=2));raise SystemExit(code)
    (S/'bootstrap_status.json').write_text(json.dumps(dict(phase='orientation_reference_audited_continue_network_research',pid=os.getpid(),network_updates=0,overall_research_complete=False),indent=2))
