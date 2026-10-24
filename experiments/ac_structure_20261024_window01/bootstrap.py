"""This chat's persistent process chain. Stops on faults; never marks research complete."""
from pathlib import Path
import datetime
import fcntl
import json
import os
import subprocess
import sys

HERE=Path(__file__).resolve().parent
def status(value):
    value['updated_at']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    p=HERE/'bootstrap_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));os.replace(t,p)

with (HERE/'bootstrap.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for stage in ['prepare','preflight','profile','run']:
        if stage=='preflight' and (HERE/'preflight.json').exists():
            assert json.loads((HERE/'preflight.json').read_text())['passed'];continue
        if stage=='profile' and (HERE/'profile/selected.json').exists():continue
        status(dict(pid=os.getpid(),phase=stage))
        code=subprocess.call([sys.executable,'-u',str(HERE/'run_window01.py'),stage],cwd=HERE.parents[1],
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1'))
        if code:
            status(dict(pid=os.getpid(),phase='fault',failed_stage=stage,returncode=code,log=str(HERE/'bootstrap.log')))
            raise SystemExit(code)
    status(dict(pid=os.getpid(),phase='round1_done_continue_research',test_used=False,
                next='Current-chat heartbeat must analyze audit and preregister next evidence-based mechanism.'))
