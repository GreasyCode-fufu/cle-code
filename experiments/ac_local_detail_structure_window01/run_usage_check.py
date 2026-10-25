"""Run the frozen preregistered CLI checker once; preserve any real failure."""
from pathlib import Path
import fcntl
import os
import runpy
import sys
import traceback

S = Path(__file__).resolve().parent
sys.path.insert(0, str(S))
import run_module_structure as m

lock = (S/'usage_check.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
try:
    m.verify()
    m.SAVE(S/'usage_check_identity.json', dict(pid=os.getpid(), source_sha256=m.sha(__file__),
           checker_sha256=m.sha(S/'check_prediction_cli.py'), started_at=m.core.timestamp(),
           owner_thread_id='01a171be-ebee-7a13-b05e-97fdb6b75587',
           policy='first fixed val actual completed weight; preregistered nativeCUDA B1; no training'))
    runpy.run_path(str(S/'check_prediction_cli.py'), run_name='__main__')
    m.SAVE(S/'usage_check_complete.json', dict(passed=True, pid=os.getpid(),
           usage_sha256=m.sha(S/'usage_smoke/independent_cli_audit.json')))
except Exception as error:
    m.SAVE(S/'usage_smoke_failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
    raise
