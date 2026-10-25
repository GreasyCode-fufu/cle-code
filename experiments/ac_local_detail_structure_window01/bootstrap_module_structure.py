from pathlib import Path
import fcntl,json,os,subprocess,sys,traceback
S=Path(__file__).resolve().parent;R=S.parents[1];sys.path.insert(0,str(S));import run_module_structure as m
def call(name,*args,cpu=False):
    env=m.engine.thread_env()
    if cpu:env['CUDA_VISIBLE_DEVICES']=''
    subprocess.run([m.PY,str(S/name),*args],cwd=R,env=env,stdin=subprocess.DEVNULL,check=True)
def main():
    lock=(S/'bootstrap.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m.SAVE(S/'bootstrap_identity.json',dict(pid=os.getpid(),source_sha256=m.sha(__file__),started_at=m.core.timestamp()))
    assert m.READ(R/'experiments/ac_midstage_focal_structure_window01/independent_audit.json')['passed']
    m.prepare()
    if not (S/'module_source_audit.json').exists():call('check_module_sources.py',cpu=True)
    if not (S/'preflight.json').exists():call('run_module_structure.py','preflight')
    if not (S/'profile/selected.json').exists():call('run_module_structure.py','profile')
    call('run_local_detail_manager.py')
    m.SAVE(S/'bootstrap_complete.json',dict(passed=True,overall_research_complete=False))
if __name__=='__main__':
    try:main()
    except Exception as exc:m.SAVE(S/'bootstrap_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
