from pathlib import Path
import hashlib,inspect,json,os,subprocess,time,traceback
R=Path('/home/login/data/cle/auto/GPT').resolve();S=Path(__file__).resolve().parent;N=R/'experiments/ac_harmonic_partition_structure_window01';PY='/home/login/miniconda3/envs/sunfuxiang/bin/python';GIT='/home/login/miniconda3/bin/git'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(n,x):(S/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def commit(message):
 while True:
  staged=subprocess.check_output([GIT,'-C',str(R),'diff','--cached','--name-only'],text=True).splitlines();safe=all(p.startswith(str(S.relative_to(R))+'/') for p in staged)
  if not (N/'independent_audit.json').exists():
   st=read(N/'status.json');safe &= bool(st.get('running'))
   for i in st.get('running',[]):
    p=N/'runs'/i['job']/'train_log.csv';meta=p.parent/'run_meta.json';epochs=len(p.read_text().splitlines())-1 if p.exists() else 0;safe &= epochs<43 and meta.exists() and time.time()-meta.stat().st_mtime>10
  if safe:break
  time.sleep(2)
 subprocess.run([GIT,'-C',str(R),'add','--',str(S.relative_to(R))],check=True)
 if subprocess.run([GIT,'-C',str(R),'diff','--cached','--quiet']).returncode:subprocess.run([GIT,'-C',str(R),'commit','-m',message],check=True)
def main():
 assert os.getuid()==1000 and not (S/'config.json').exists();import torchvision.models.swin_transformer as tv
 runtime=Path(inspect.getfile(tv));assert sha(runtime)=='9ec73f2333108389e152fda346b6f195ebced02e9ced367cef60f405093b40b1';(S/'upstream/runtime_torchvision_swin.py').write_bytes(runtime.read_bytes());(S/'upstream/LICENSE_TORCHVISION').write_bytes((R/'experiments/ac_deform_geometry_prototype_window01/upstream/LICENSE_TORCHVISION').read_bytes());(S/'c4_moment_stem.py').write_bytes((R/'models/optimize/ac_module_window01/c4_moment_stem.py').read_bytes());assert sha(S/'c4_moment_stem.py')=='8cd90f0659a7a852fa16c7a848bca0c416f24913c3548ee6e6141792cf429fdd'
 provenance=read(S/'upstream/provenance.json')
 for row in provenance['files']:
  raw=(S/'upstream'/row['saved_as']).read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['blob']
 Q=R/'experiments/ac_qualification_20261024_window01';asset=read(Q/'pretrained_manifest.json')['assets']['swin_t'];assert sha(R/asset['path'])==asset['sha256'];(S/'.gitignore').write_text('exports/\n__pycache__/\n*.pyc\n')
 files=[S/n for n in ['bootstrap.py','check_merge_cpu.py','merge_contrast.py','c4_moment_stem.py','PROTO_PROTOCOL_ZH.md']]+[p for p in (S/'upstream').rglob('*') if p.is_file()];locked=[Q/'config.json',Q/'pretrained_manifest.json',R/'experiments/ac_backbone_transfer_evidence_window01/evidence.json',N/'config.json']
 save('config.json',dict(created_at=time.strftime('%Y-%m-%d %H:%M:%S%z'),sources={str(p.relative_to(R)):sha(p) for p in files},locked={str(p.relative_to(R)):sha(p) for p in locked},runtime_source=str(runtime),runtime_sha256=sha(runtime),public_asset_sha256=asset['sha256'],eps=dict(noaffine_LN=1e-5,spatial_stat=1e-6),controls=['local_std','mean','global_std','with_without_retained_Mom'],scope='CPU synthetic/public compatibility only, next formal decision after full POWER controls',CLE_images=0,CLE_training=0,new_test_inference=0,overall_research_complete=False));save('identity.json',dict(pid=os.getpid(),source_sha256=sha(__file__)));commit('research: preregister retained-moment Swin merge spatial-contrast numerical prototype and official-layer controls')
 subprocess.run([PY,str(S/'check_merge_cpu.py')],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,check=True)
 for p,h in read(S/'config.json')['sources'].items():assert sha(R/p)==h
 save('complete.json',dict(passed=read(S/'numeric_preflight_cpu.json')['passed'],config_sha256=sha(S/'config.json'),proof_sha256=sha(S/'numeric_preflight_cpu.json'),CLE_training=0,CLE_images=0,new_test_inference=0,overall_research_complete=False));commit('test: verify normalized merge contrast Haar oracle official parity all gradients and public Swin384 pairing')
if __name__=='__main__':
 try:main()
 except Exception as e:save('failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
