"""Preserve original failed runner/config; repair only str-to-Path registration hash."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,traceback
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import bootstrap as old
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert os.getuid()==1000 and not (S/'complete.json').exists();cfg=old.read(S/'config.json');failure=old.read(S/'failure.json');assert 'read_bytes' in failure['error']
 for group in ['sources','locked']:
  for p,h in cfg[group].items():assert sha(old.R/p)==h,p
 old.save('startup_repair_v2.json',dict(reason='initial bootstrap passed __file__ string to Path-only digest before identity/Git/CPU execution; no numerical test or CLE training ran',original_failure_sha256=sha(S/'failure.json'),original_config_sha256=sha(S/'config.json'),repair_source_sha256=sha(__file__),original_sources_unchanged=True,new_training=0))
 old.save('identity_v2.json',dict(pid=os.getpid(),source_sha256=sha(__file__)));old.commit('research: preserve harmonic prototype preregistration and failed hash receipt; register isolated runner repair')
 subprocess.run([old.PY,str(S/'check_harmonic_cpu.py')],cwd=old.R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,check=True)
 for p,h in cfg['sources'].items():assert sha(old.R/p)==h
 old.save('complete.json',dict(passed=old.read(S/'numeric_preflight_cpu.json')['passed'],config_sha256=sha(S/'config.json'),proof_sha256=sha(S/'numeric_preflight_cpu.json'),repair_registration_sha256=sha(S/'startup_repair_v2.json'),original_failure_retained=True,CLE_training=0,CLE_images=0,overall_research_complete=False));old.commit('test: verify C4 power Parseval official magnitude gradients group relation and public initialization after runner-only repair')
if __name__=='__main__':
 try:main()
 except Exception as e:old.save('repair_failure_v2.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
