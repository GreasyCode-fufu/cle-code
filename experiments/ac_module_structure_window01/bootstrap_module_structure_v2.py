"""Own-window continuation: preserve previous live models, then preflight/profile/train/audit."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback
S=Path(__file__).resolve().parent;R=S.parents[1];sys.path.insert(0,str(S))
import run_module_structure as m

def call(*args,cpu=False):
    env=m.engine.thread_env()
    if cpu:env['CUDA_VISIBLE_DEVICES']=''
    subprocess.run([m.PY,*map(str,args)],cwd=R,env=env,stdin=subprocess.DEVNULL,check=True)

def main():
    m.SAVE(S/'bootstrap_identity_v2.json',dict(pid=os.getpid(),source_sha256=m.sha(__file__),started_at=m.core.timestamp()))
    m.prepare()
    revision=m.READ(S/"cpu_audit_revision_v2.json");assert revision["original_config_sha256"]==m.sha(S/"config.json") and all(m.sha(S/p)==h for p,h in revision["sources"].items())
    m.commit("audit: preserve ECA machine-rounding diagnosis and freeze CPU checker revision2; model sources unchanged")
    if not (S/'module_source_audit.json').exists():call(S/'check_module_sources_v2.py',cpu=True)
    else:assert m.READ(S/'module_source_audit.json')['passed']
    while True:
        identity=m.READ(m.F/'queue_manager_identity.json');p=Path('/proc')/str(identity['pid'])/'cmdline';live=p.exists() and b'manage_field_queue_v2.py' in p.read_bytes()
        status=m.READ(m.F/'queue_status_v2.json')
        m.SAVE(S/'bootstrap_status_v2.json',dict(phase='waiting_previous_valid_live_models' if live else 'checking_previous_terminal',previous_status=status,updated_at=m.core.timestamp(),overall_research_complete=False))
        if not live:
            assert (m.F/'independent_audit.json').exists() and m.READ(m.F/'independent_audit.json')['passed'],'previous own queue ended without final audit; inspect logs and surviving workers first'
            break
        # No arbitrary mixed oversubscription: previous family active GPU98-100%,
        # measured concurrency slower. Prepare on CPU and queue real next methods.
        time.sleep(20)
    if not (S/'preflight.json').exists():call(S/'run_module_structure.py','preflight')
    else:assert m.READ(S/'preflight.json')['passed']
    if not (S/'profile/selected.json').exists():call(S/'run_module_structure.py','profile')
    call(S/'run_module_structure.py','run')
    m.SAVE(S/'bootstrap_status_v2.json',dict(phase='round_audited_continue_research',overall_research_complete=False,updated_at=m.core.timestamp()))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        m.SAVE(S/'bootstrap_failure_v2.json',dict(error=repr(exc),traceback=traceback.format_exc(),updated_at=m.core.timestamp(),action='inspect exact own logs and surviving process identities; valid complete checkpoints reused'));raise
