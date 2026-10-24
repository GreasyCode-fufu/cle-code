from pathlib import Path
import os,subprocess,sys
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
m.verify();m.SAVE(S/'manager_entry_identity_v2.json',dict(pid=os.getpid(),source_sha256=m.sha(__file__),started_at=m.core.timestamp()))
if not (S/'runtime_sampler_identity.json').exists():
    log=(S/'runtime_sampler.log').open('a',buffering=1);p=subprocess.Popen([m.PY,str(S/'continuous_runtime_sampler.py')],cwd=m.R,env=m.engine.thread_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
m.run()
