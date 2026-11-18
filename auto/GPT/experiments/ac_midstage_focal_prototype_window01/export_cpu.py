from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile
R=Path('/home/login/data/cle/auto/GPT').resolve();S=R/'experiments/ac_midstage_focal_prototype_window01';M=R/'models/optimize/ac_midstage_focal_prototype_window01'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.getuid()==1000 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
cfg=json.loads((S/'config.json').read_text());proof=json.loads((S/'numeric_preflight_cpu.json').read_text());done=json.loads((S/'complete.json').read_text());identity=json.loads((S/'cpu_identity.json').read_text())
assert not Path('/proc',str(identity['pid'])).exists()
assert done['passed'] and proof['passed'] and done['proof_sha256']==sha(S/'numeric_preflight_cpu.json') and done['source_config_sha256']==sha(S/'config.json')
assert all(sha(R/p)==h for kind in ['sources','locked'] for p,h in cfg[kind].items())
assert not (S/'failure.json').exists() and len(proof['public_initializations'])==21
git='/home/login/miniconda3/bin/git'
assert not subprocess.check_output([git,'-C',str(R),'diff','--cached','--name-only'],text=True).strip()
subprocess.run([git,'-C',str(R),'add','--',str(S.relative_to(R)),str(M.relative_to(R))],check=True)
if subprocess.run([git,'-C',str(R),'diff','--cached','--quiet']).returncode:
 subprocess.run([git,'-C',str(R),'commit','-m','results: preserve midstage focal contrast CPU oracles and public initialization proofs'],check=True)
data={};manifest=[]
for root in [S,M]:
 for p in sorted(root.rglob('*')):
  if not p.is_file() or any(v in p.parts for v in ['exports','__pycache__']) or p.suffix in ['.pyc','.tmp','.lock']:continue
  raw=p.read_bytes();name=str(p.relative_to(R));data[name]=raw;manifest.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
out=S/'exports';out.mkdir(exist_ok=True);target=out/'midstage_focal_cpu_evidence.tar.gz'
with tarfile.open(target,'w:gz') as t:
 for name,raw in data.items():
  item=tarfile.TarInfo(name);item.size=len(raw);item.uid=item.gid=1000;item.uname=item.gname='login';t.addfile(item,io.BytesIO(raw))
 raw=json.dumps(dict(files=manifest,CPU_only=True,CLE_images=0,overall_research_complete=False),ensure_ascii=False,indent=2).encode();item=tarfile.TarInfo('EVIDENCE_MANIFEST.json');item.size=len(raw);t.addfile(item,io.BytesIO(raw))
receipt=dict(path=str(target.relative_to(R)),sha256=sha(target),bytes=target.stat().st_size,files=len(manifest),git=subprocess.check_output([git,'-C',str(R),'rev-parse','--short','HEAD'],text=True).strip())
(out/'export_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
