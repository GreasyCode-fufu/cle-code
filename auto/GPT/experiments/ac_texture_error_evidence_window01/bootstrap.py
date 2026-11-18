from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile,traceback
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import texture_evidence as e
def main():
 e.save(S/'identity.json',dict(pid=os.getpid(),source_sha256=e.sha(__file__),started_at=e.stamp()));e.prepare();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');py='/home/login/miniconda3/envs/sunfuxiang/bin/python'
 if not (S/'result.json').exists():subprocess.run([py,str(S/'analyze_texture_evidence.py')],cwd=e.R,env=env,stdin=subprocess.DEVNULL,check=True)
 if not (S/'independent_audit.json').exists():subprocess.run([py,str(S/'audit_texture_evidence.py')],cwd=e.R,env=env,stdin=subprocess.DEVNULL,check=True)
 assert e.read(S/'independent_audit.json')['passed'];e.save(S/'complete.json',dict(passed=True,config_sha256=e.sha(S/'config.json'),result_sha256=e.sha(S/'result.json'),audit_sha256=e.sha(S/'independent_audit.json'),neural_training=0,test_images=0,overall_research_complete=False));e.commit('results: preserve completed audited CPU texture-error mechanism evidence')
 data={};manifest=[]
 for p in sorted(S.rglob('*')):
  if not p.is_file() or any(x in p.parts for x in ['exports','__pycache__']) or p.suffix in ['.pyc','.log','.lock']:continue
  raw=p.read_bytes();name=str(p.relative_to(e.R));data[name]=raw;manifest.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
 out=S/'exports';out.mkdir(exist_ok=True);target=out/'texture_error_evidence.tar.gz'
 with tarfile.open(target,'w:gz') as tar:
  for name,raw in data.items():
   item=tarfile.TarInfo(name);item.size=len(raw);item.uid=item.gid=1000;item.uname=item.gname='login';item.mode=0o644;tar.addfile(item,io.BytesIO(raw))
  raw=json.dumps(dict(files=manifest,neural_training=0,test_images=0),ensure_ascii=False,indent=2).encode();item=tarfile.TarInfo('EVIDENCE_MANIFEST.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 e.save(out/'export_receipt.json',dict(path=str(target.relative_to(e.R)),sha256=e.sha(target),bytes=target.stat().st_size,files=len(manifest)));print(json.dumps(e.read(out/'export_receipt.json')),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:e.save(S/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),action='inspect precise own logs; no automatic retraining'));raise
