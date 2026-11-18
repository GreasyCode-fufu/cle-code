"""Change-triggered coordinator around immutable v2 inference; avoids repeated no-op summaries."""
from pathlib import Path
import fcntl,json,os,sys,time,traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_test_feedback_v2 as t

def live(pid,needle):
    try:return needle in (Path('/proc')/str(pid)/'cmdline').read_bytes().decode(errors='replace')
    except (FileNotFoundError,PermissionError,ProcessLookupError):return False

def main():
    with (HERE/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        t.save(HERE/'coordinator_identity.json',dict(pid=os.getpid(),source=str(Path(__file__).resolve()),source_sha256=t.sha(__file__),started_at=t.core.timestamp()))
        previous=None;counter=0
        while True:
            q=t.read(t.Q/'status.json')
            key=t.signature(dict(completed_new=q.get('completed_new'),qualification_phase=q.get('phase'),qualification_selected=t.sha(t.Q/'selected_baseline.json') if (t.Q/'selected_baseline.json').exists() else None,
                qualification_audited=(t.Q/'independent_audit.json').exists() and t.read(t.Q/'independent_audit.json')['passed']))
            if key!=previous:
                # This is inference only: valid completed predictions are reused by v2 fingerprint.
                t.run(watch=False);previous=key
            report=t.read(HERE/'test_results.json')
            complete=report['complete_method_groups']==15 and report['unique_audited_checkpoints']==42 and report['primary_baseline'] is not None
            t.save(HERE/'coordinator_status.json',dict(phase='feedback_completed_continue_structure' if complete else 'waiting_for_new_complete_paired_method_or_baseline_audit',
                pid=os.getpid(),updated_at=t.core.timestamp(),groups=report['complete_method_groups'],unique_audited_checkpoints=report['unique_audited_checkpoints'],
                primary_baseline=report['primary_baseline'],overall_research_complete=False,qualification_phase=q.get('phase')))
            if complete:
                t.safe_commit('research: finish all qualified test feedback and preserve fixed baseline for next mechanism')
                return
            if counter%1==0:t.first.append_resource(HERE/'coordinator_resources.csv',t.resources())
            counter+=1;time.sleep(2)

if __name__=='__main__':
    try:main()
    except Exception:
        t.save(HERE/'coordinator_fault.json',dict(timestamp=t.core.timestamp(),traceback=traceback.format_exc()))
        raise
