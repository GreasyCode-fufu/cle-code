from pathlib import Path
import json,os,subprocess,sys,time,traceback
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
def main():
    m.SAVE(S/'direct_control_finisher_identity.json',dict(pid=os.getpid(),source_sha256=m.sha(__file__),started_at=m.core.timestamp()))
    while not (S/'independent_audit.json').exists():
        if (S/'bootstrap_failure.json').exists():raise RuntimeError('own bootstrap failed; inspect exact logs, no training restart performed')
        time.sleep(5)
    assert m.READ(S/'independent_audit.json')['passed']
    while True:
        p=Path('/proc')/str(m.READ(S/'manager_entry_identity_v2.json')['pid'])/'cmdline'
        if not p.exists() or b'run_covariance_manager.py' not in p.read_bytes():break
        time.sleep(2)
    subprocess.run([m.PY,str(S/'compare_covariance_controls.py')],cwd=m.R,env=dict(m.engine.thread_env(),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,check=True)
    m.SAVE(S/'direct_control_finisher_complete.json',dict(passed=True,comparison_sha256=m.sha(S/'direct_control_comparisons.json'),overall_research_complete=False))
    m.commit('audit: preserve completed channel-covariance direct-control finisher provenance')
if __name__=='__main__':
    try:main()
    except Exception as exc:m.SAVE(S/'direct_control_finisher_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
