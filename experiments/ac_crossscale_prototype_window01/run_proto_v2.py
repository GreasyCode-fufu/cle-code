from pathlib import Path
import json,os,subprocess,sys,traceback
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import bootstrap_proto as old
def main():
    old.save(S/'identity_v2.json',dict(pid=os.getpid(),source_sha256=old.sha(__file__)))
    rev=json.loads((S/'cpu_revision_v2.json').read_text());assert rev['original_config_sha256']==old.sha(S/'config.json') and all(old.sha(S/p)==h for p,h in rev['sources'].items())
    old.quiet_commit('audit: preserve CPU prototype import-path failure and freeze checker revision2; operator source unchanged')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    subprocess.run(['/home/login/miniconda3/envs/sunfuxiang/bin/python',str(S/'check_crossscale_v2.py')],cwd=old.R,env=env,stdin=subprocess.DEVNULL,check=True)
    assert json.loads((S/'numeric_preflight_cpu.json').read_text())['passed']
    old.save(S/'complete_v2.json',dict(passed=True,real_neural_training=False,preflight_sha256=old.sha(S/'numeric_preflight_cpu.json'),overall_research_complete=False))
    old.quiet_commit('test: pass bounded cross-scale CPU operators and exact public moment-model initialization')
if __name__=='__main__':
    try:main()
    except Exception as e:old.save(S/'failure_v2.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
