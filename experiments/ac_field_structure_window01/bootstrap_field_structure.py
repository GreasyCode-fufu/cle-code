"""Wait on actual own prior processes, then freeze/precheck/profile/train/audit field round."""
from pathlib import Path
import datetime,fcntl,json,os,subprocess,time
S=Path(__file__).resolve().parent;R=S.parents[1];Q=R/'experiments/ac_qualification_20261024_window01';T=R/'experiments/ac_test_feedback_window01'
PY='/home/login/miniconda3/envs/sunfuxiang/bin/python'
def read(p):return json.loads(p.read_text())
def save(v):
    v.update(pid=os.getpid(),updated_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    p=S/'bootstrap_status.json';part=p.with_suffix('.tmp');part.write_text(json.dumps(v,indent=2));os.replace(part,p)
def live(pid,needle):
    try:return needle in (Path('/proc')/str(pid)/'cmdline').read_bytes().decode(errors='replace')
    except (FileNotFoundError,PermissionError,ProcessLookupError):return False
def call(stage):
    save(dict(phase=stage));code=subprocess.call([PY,'-u',str(S/'run_field_structure.py'),stage],cwd=R,
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    if code:save(dict(phase='fault',failed_stage=stage,returncode=code));raise SystemExit(code)
with (S/'bootstrap.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        q=read(Q/'status.json');t=read(T/'coordinator_status.json');qp=read(Q/'pipeline_pid.json')['pid'];tp=read(T/'coordinator_identity.json')['pid']
        qlive=live(qp,'ac_qualification_20261024_window01/run_qualification.py run');tlive=live(tp,'ac_test_feedback_window01/coordinate_feedback.py')
        # Also wait for qualification's own bootstrap to finish its final commit.
        qb=read(Q/'bootstrap_status.json');qblive=live(qb['pid'],'ac_qualification_20261024_window01/bootstrap_qualification.py')
        if not qlive and not tlive and not qblive:
            assert read(Q/'independent_audit.json')['passed'] and read(Q/'selected_baseline.json')['frozen']
            assert t['phase']=='feedback_completed_continue_structure' and t['unique_audited_checkpoints']==42
            assert not (T/'coordinator_fault.json').exists()
            break
        if q.get('failures') or (T/'coordinator_fault.json').exists():save(dict(phase='prior_fault_needs_diagnosis',qualification=q));raise SystemExit(1)
        save(dict(phase='waiting_verified_qualification_and_test_feedback',qualification_alive=qlive,qualification_bootstrap_alive=qblive,
            feedback_alive=tlive,qualification_completed=q.get('completed_new'),feedback_checkpoints=t.get('unique_audited_checkpoints')))
        time.sleep(10)
    for stage in ['prepare','preflight','profile','run']:
        if stage=='preflight' and (S/'preflight.json').exists():assert read(S/'preflight.json')['passed'];continue
        if stage=='profile' and (S/'profile/selected.json').exists():continue
        call(stage)
    save(dict(phase='field_round_done_continue_research',overall_research_complete=False))
