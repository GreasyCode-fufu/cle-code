"""Own-window feedback runner; no training restart or foreign process control."""
from pathlib import Path
import json,os,subprocess,time
HERE=Path(__file__).resolve().parent
PY='/home/login/miniconda3/envs/sunfuxiang/bin/python'
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (HERE/'runner_v2.log').open('a',buffering=1) as out:
    p=subprocess.Popen([PY,'-u',str(HERE/'run_test_feedback_v2.py'),'--watch'],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,env=env)
    (HERE/'runner_identity_v2.json').write_text(json.dumps(dict(pid=p.pid,bootstrap_pid=os.getpid(),source=str(HERE/'run_test_feedback_v2.py'),started_at=time.time()),indent=2))
    code=p.wait()
    (HERE/'runner_exit_v2.json').write_text(json.dumps(dict(returncode=code,finished_at=time.time()),indent=2))
    if code:raise SystemExit(code)
