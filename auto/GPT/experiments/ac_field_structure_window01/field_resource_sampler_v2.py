"""Continuous resource sampling for adopted training, runtime profiling and audits."""
from pathlib import Path
import os,sys,time
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import run_field_structure as f
def own_live():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if S.name in cmd and any(n in cmd for n in ['run_field_structure.py','manage_field_queue_v2.py','audit_field_structure.py','bench_field_runtime.py']):return True
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return False
f.SAVE(S/'resource_sampler_identity_v2.json',dict(pid=os.getpid(),source_sha256=f.sha(__file__),sample_seconds=2))
while own_live():
    f.engine.append_resource(S/'continuous_resources_v2.csv',f.engine.snapshot());time.sleep(2)
