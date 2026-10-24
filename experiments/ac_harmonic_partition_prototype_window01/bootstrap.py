from pathlib import Path
import hashlib,json,os,subprocess,time,traceback
R=Path('/home/login/data/cle/auto/GPT').resolve();S=Path(__file__).resolve().parent;P=R/'experiments/ac_patient_alignment_structure_window01';PY='/home/login/miniconda3/envs/sunfuxiang/bin/python';GIT='/home/login/miniconda3/bin/git'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(n,x):(S/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def commit(message):
 while True:
  staged=subprocess.check_output([GIT,'-C',str(R),'diff','--cached','--name-only'],text=True).splitlines();safe=all(p.startswith(str(S.relative_to(R))+'/') for p in staged)
  identity=P/'manager_entry_identity_v2.json'
  if identity.exists() and not (P/'independent_audit.json').exists():
   proc=Path('/proc')/str(read(identity)['pid'])/'cmdline'
   if proc.exists() and b'run_patient_manager.py' in proc.read_bytes():
    st=read(P/'status.json');safe &= bool(st.get('running'))
    for item in st.get('running',[]):
     log=P/'runs'/item['job']/'train_log.csv';meta=log.parent/'run_meta.json';epochs=len(log.read_text().splitlines())-1 if log.exists() else 0;safe &= epochs<43 and meta.exists() and time.time()-meta.stat().st_mtime>10
  if safe:break
  time.sleep(2)
 subprocess.run([GIT,'-C',str(R),'add','--',str(S.relative_to(R))],check=True)
 if subprocess.run([GIT,'-C',str(R),'diff','--cached','--quiet']).returncode:subprocess.run([GIT,'-C',str(R),'commit','-m',message],check=True)
def main():
 assert os.getuid()==1000 and S.stat().st_uid==1000;assert not (S/'config.json').exists(),'prototype already registered; inspect exact logs instead of overwriting'
 (S/'c4_moment_stem.py').write_bytes((R/'models/optimize/ac_module_window01/c4_moment_stem.py').read_bytes());assert sha(S/'c4_moment_stem.py')=='8cd90f0659a7a852fa16c7a848bca0c416f24913c3548ee6e6141792cf429fdd'
 (S/'.gitignore').write_text('exports/\n__pycache__/\n*.pyc\n');provenance=read(S/'upstream/provenance.json')
 for row in provenance['files']:
  b=(S/'upstream'/row['saved_as']).read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['sha']
 files=[S/n for n in ['bootstrap.py','harmonic_partition.py','check_harmonic_cpu.py','c4_moment_stem.py','PROTO_PROTOCOL_ZH.md']]+[p for p in (S/'upstream').rglob('*') if p.is_file()];locked=[R/'experiments/ac_module_structure_window01/config.json',R/'experiments/ac_module_structure_window01/independent_audit.json',P/'config.json',R/'experiments/ac_dataset_v1/dataset_manifest.json']
 save('config.json',dict(created_at=time.strftime('%Y-%m-%d %H:%M:%S%z'),sources={str(p.relative_to(R)):sha(p) for p in files},locked={str(p.relative_to(R)):sha(p) for p in locked},operator='bounded C4 first/second power contrast complementary to frozen mean/std',controls=['noncyclic permutation','same-branch duplicate std','official rawH2 magnitude'],kind='CPU numerical prototype only',public_initialization_only=True,CLE_images=0,CLE_training=0,test_predictions=0,formal_next_round_not_started=True,overall_research_complete=False));save('identity.json',dict(pid=os.getpid(),source_sha256=sha(__file__)));commit('research: preregister C4 power-shape complement numerical prototype and faithful magnitude control')
 subprocess.run([PY,str(S/'check_harmonic_cpu.py')],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,check=True)
 cfg=read(S/'config.json');assert all(sha(R/p)==h for p,h in cfg['sources'].items());save('complete.json',dict(passed=read(S/'numeric_preflight_cpu.json')['passed'],config_sha256=sha(S/'config.json'),proof_sha256=sha(S/'numeric_preflight_cpu.json'),CLE_training=0,CLE_images=0,overall_research_complete=False));commit('test: verify C4 power Parseval official magnitude gradients group relation and public initialization')
if __name__=='__main__':
 try:main()
 except Exception as e:save('failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
