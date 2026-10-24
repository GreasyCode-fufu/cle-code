"""New postprocessing wrapper; immutable training functions/config remain unchanged."""
from pathlib import Path
import importlib.util,sys,os
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import run_module_structure as m
cfg=m.READ(S/'postprocessing_revision_v2.json');assert cfg['original_config_sha256']==m.sha(S/'config.json')
assert all(m.sha(S/p)==h for p,h in cfg['sources'].items());m.verify()
spec=importlib.util.spec_from_file_location('audit_module_structure',S/'audit_module_structure_v2.py')
audit=importlib.util.module_from_spec(spec);sys.modules['audit_module_structure']=audit;spec.loader.exec_module(audit)
m.SAVE(S/'manager_entry_identity_v2.json',dict(pid=os.getpid(),source_sha256=m.sha(__file__),postprocessing_config_sha256=m.sha(S/'postprocessing_revision_v2.json'),started_at=m.core.timestamp()))
m.run()
