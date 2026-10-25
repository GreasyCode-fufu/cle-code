"""Register and launch one documentation, one actual-CLI and one CPU exporter helper."""
from pathlib import Path
import csv,json,os,subprocess,sys,time,traceback
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
F=m.R/'experiments/ac_spatial_odd_final_report_window01'

def main():
    m.verify();assert os.getuid()==1000 and not (S/'supplemental_registration.json').exists()
    for p in [S/'document_odd_start.py',S/'check_prediction_cli.py',F/'export_odd_final.py']:
        assert p.stat().st_uid==1000;compile(p.read_text(),str(p),'exec')
    while True:
        assert not (S/'bootstrap_failure.json').exists(),'own pipeline fault; do not restart training'
        st=m.READ(S/'status.json') if (S/'status.json').exists() else {};safe=bool(st.get('running'))
        for i in st.get('running',[]):
            proc=Path('/proc')/str(i['pid']);log=S/'runs'/i['job']/'train_log.csv';meta=log.parent/'run_meta.json'
            rows=list(csv.DictReader(log.open())) if log.exists() else []
            safe &= proc.exists() and i['job'].encode() in (proc/'cmdline').read_bytes() and len(rows)<43 and meta.exists() and time.time()-meta.stat().st_mtime>10
        if safe:break
        time.sleep(2)
    sources={str(p.relative_to(m.R)):m.sha(p) for p in [Path(__file__),S/'document_odd_start.py',S/'check_prediction_cli.py',F/'export_odd_final.py']}
    registration=dict(registered_at=m.core.timestamp(),source_config_sha256=m.sha(S/'config.json'),sources=sources,CUDA_training_sources_unchanged=True,new_training=0,overall_research_complete=False)
    m.SAVE(S/'supplemental_registration.json',registration);m.SAVE(F/'exporter_registration.json',registration)
    allowed=[str(p.relative_to(m.R)) for p in [S,m.M,F]]
    staged=subprocess.check_output([m.core.GIT,'-C',str(m.R),'diff','--cached','--name-only'],text=True).splitlines()
    assert all(any(p.startswith(a+'/') for a in allowed) for p in staged),'unrelated staged changes left untouched'
    subprocess.run([m.core.GIT,'-C',str(m.R),'add','--',*allowed],check=True)
    if subprocess.run([m.core.GIT,'-C',str(m.R),'diff','--cached','--quiet']).returncode:subprocess.run([m.core.GIT,'-C',str(m.R),'commit','-m','docs: register spatial-odd actual-start actual-checkpoint CLI and complete-round CPU exporter'],check=True)
    helpers=[]
    for path,logname,cpu,failure in [(S/'document_odd_start.py','document_start.log',True,'document_start_failure.json'),(S/'check_prediction_cli.py','usage_smoke.log',False,'usage_smoke_failure.json'),(F/'export_odd_final.py','export.log',True,'export_failure.json')]:
        env=m.engine.thread_env()
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        else:env.pop('CUDA_VISIBLE_DEVICES',None)
        log=path.parent/logname
        # Wrapper preserves any concrete helper failure even when its script has no catch.
        wrapper="import runpy,pathlib,json,traceback,sys; p=pathlib.Path(sys.argv[1]);\ntry: runpy.run_path(str(p),run_name='__main__')\nexcept Exception as e: p.with_name(sys.argv[2]).write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2)); raise"
        with log.open('a',buffering=1) as stream:
            p=subprocess.Popen([m.PY,'-c',wrapper,str(path),failure],cwd=m.R,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        ident=dict(pid=p.pid,source_sha256=m.sha(path),started_at=m.core.timestamp(),CPU_only=cpu,new_training=0)
        m.SAVE(path.parent/(path.stem+'_launch_identity.json'),ident);helpers.append(dict(path=str(path.relative_to(m.R)),**ident))
    m.SAVE(S/'supplement_helpers_launched.json',dict(passed=True,helpers=helpers));print(json.dumps(helpers),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:m.SAVE(S/'supplement_launcher_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise


