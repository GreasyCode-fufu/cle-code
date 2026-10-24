from pathlib import Path
import fcntl,json,os,subprocess,sys,time
S=Path(__file__).resolve().parent;R=S.parents[1];sys.path.insert(0,str(R/'experiments/ac_patient_mechanism_evidence_window01'))
import patient_evidence as guard
guard.S=S
PY='/home/login/miniconda3/envs/sunfuxiang/bin/python'
with (S/'bootstrap.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for p in S.glob('*.py'):compile(p.read_text(),str(p),'exec')
    guard.commit('research: register identical C4 controls on frozen B0 and DINO, preserving primary baseline')
    for control in ['B0','DINO']:
        for stage in ['prepare','preflight','evaluate','analyze']:
            (S/'bootstrap_status.json').write_text(json.dumps(dict(phase=stage,control=control,pid=os.getpid(),updated_at=time.time()),indent=2))
            code=subprocess.call([PY,'-u',str(S/'run_c4_budget_control.py'),'--control',control,stage],cwd=R,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            if code:
                (S/'bootstrap_status.json').write_text(json.dumps(dict(phase='fault',control=control,stage=stage,returncode=code,pid=os.getpid()),indent=2));raise SystemExit(code)
    (S/'bootstrap_status.json').write_text(json.dumps(dict(phase='budget_controls_complete_continue_network_research',pid=os.getpid(),overall_research_complete=False),indent=2))
