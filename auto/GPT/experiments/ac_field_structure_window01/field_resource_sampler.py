"""Continuous2s GPU/CPU/core/RAM sidecar, including audits between training tasks."""
from pathlib import Path
import json,os,sys,time
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import run_field_structure as f

def own_live():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            cmd=(p/'cmdline').read_bytes().decode(errors='replace')
            if S.name in cmd and any(x in cmd for x in ['bootstrap_field_structure.py','run_field_structure.py','audit_field_structure.py']):return True
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return False

(S/'resource_sampler_identity.json').write_text(json.dumps(dict(pid=os.getpid(),sample_seconds=2,source_sha256=f.sha(__file__)),indent=2))
while own_live():
    phase=f.READ(S/'bootstrap_status.json').get('phase','unknown') if (S/'bootstrap_status.json').exists() else 'launch'
    f.engine.append_resource(S/'continuous_resources.csv',dict(**f.engine.snapshot(),bootstrap_phase=phase))
    time.sleep(2)
