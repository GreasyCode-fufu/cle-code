"""Prepare assets while prior runs, then require its audited terminal success."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time,datetime
S=Path(__file__).resolve().parent;R=S.parents[1];P=R/'experiments/ac_structure_20261024_window01'
PY='/home/login/miniconda3/envs/sunfuxiang/bin/python'

def save(x):
 x.update(pid=os.getpid(),updated_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
 p=S/'bootstrap_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)

def own_live(pid,expected):
 p=Path('/proc')/str(pid)/'cmdline'
 if not p.exists():return False
 command=p.read_bytes().replace(b'\0',b' ').decode()
 if expected not in command:return False
 return True

def call(stage):
 save(dict(phase=stage))
 code=subprocess.call([PY,'-u',str(S/'run_qualification.py'),stage],cwd=R,
  env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1'))
 if code:
  save(dict(phase='fault',failed_stage=stage,returncode=code,log=str(S/'bootstrap.log')));raise SystemExit(code)

with (S/'bootstrap.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 call('assets')
 while True:
  state=json.loads((P/'bootstrap_status.json').read_text())
  if state.get('phase')=='fault':
   save(dict(phase='prior_fault',prior=state));raise SystemExit('Prior fault; inspect its actual logs and own live processes')
  bp=json.loads((P/'launch.json').read_text())['pid'];qp=json.loads((P/'pipeline_pid.json').read_text())['pid']
  lives=[dict(pid=bp,alive=own_live(bp,str(P/'bootstrap.py'))),dict(pid=qp,alive=own_live(qp,'ac_structure_20261024_window01/run_window01.py run'))]
  audit=P/'independent_audit.json'
  if not any(x['alive'] for x in lives):
   if not audit.exists():
    save(dict(phase='prior_process_terminal_without_audit',prior_processes=lives));raise SystemExit('Prior stopped without independent audit')
   assert json.loads(audit.read_text())['passed']
   break
  save(dict(phase='waiting_verified_prior_process_and_audit',prior_processes=lives,prior_state=state.get('phase')))
  time.sleep(10)
 pairs=json.loads((P/'paired_comparisons.json').read_text())
 if pairs['relational_vs_baseline']['strong_dev_signal']:
  save(dict(phase='alternate_lane_not_triggered',reason='DINO primary passed; continue its evidence instead of automatically changing baseline'))
  raise SystemExit(0)
 for stage in ['prepare','preflight','profile','run']:
  if stage=='preflight' and (S/'preflight.json').exists():assert json.loads((S/'preflight.json').read_text())['passed'];continue
  if stage=='profile' and (S/'profile/selected.json').exists():continue
  call(stage)
 save(dict(phase='qualification_done_continue_structure_design',test_used=False))
