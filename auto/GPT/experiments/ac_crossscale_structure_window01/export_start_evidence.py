from pathlib import Path
import hashlib,io,json,os,sys,tarfile,time,csv,subprocess
S=Path(__file__).resolve().parent;R=S.parents[1];sys.path.insert(0,str(S));import run_module_structure as m
m.verify();assert m.READ(S/'usage_smoke/independent_cli_audit.json')['passed'];(S/'.gitignore').write_text('exports/\n__pycache__/\n*.pyc\n')
data={};files=[]
for p in sorted(S.rglob('*'))+sorted(m.M.rglob('*')):
    if not p.is_file() or any(part in p.parts for part in ['exports','__pycache__','runs','logs']) or p.suffix in ['.lock','.log','.pyc','.tmp','.jsonl']:continue
    if p.parent==S and (p.name in ['resources.csv','status.json','continuous_runtime_resources.csv','START_EVIDENCE_RECEIPT.json'] or p.name.startswith('bootstrap_') and p.suffix=='.json'):continue
    # Resource timelines remain server-side; this package freezes completed setup/proofs only.
    if p.name=='resources.csv':continue
    raw=p.read_bytes();name=str(p.relative_to(R));data[name]=raw;files.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
out=S/'exports';out.mkdir(exist_ok=True);target=out/'crossscale_start_evidence.tar.gz'
with tarfile.open(target,'w:gz') as tar:
    for name,raw in data.items():
        info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644;info.uid=info.gid=1000;info.uname=info.gname='login';tar.addfile(info,io.BytesIO(raw))
    raw=json.dumps(dict(files=files,scope='frozen source/config/preregistration/numerical proofs/completed throughput profiles/actual checkpoint usage; formal training ongoing',weight_binaries_on_server=True),ensure_ascii=False,indent=2).encode();info=tarfile.TarInfo('EVIDENCE_MANIFEST.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
receipt=dict(path=str(target.relative_to(R)),sha256=m.sha(target),bytes=target.stat().st_size,files=len(files),config_sha256=m.sha(S/'config.json'),training_ongoing=True,not_final_effectiveness_result=True)
m.SAVE(S/'START_EVIDENCE_RECEIPT.json',receipt)
while True:
    state=m.READ(S/'status.json');safe=bool(state.get('running'))
    for item in state.get('running',[]):
        log=S/'runs'/item['job']/'train_log.csv';meta=log.parent/'run_meta.json';rows=list(csv.DictReader(log.open())) if log.exists() else [];safe &= len(rows)<45 and meta.exists() and time.time()-meta.stat().st_mtime>10
    if safe or (S/'independent_audit.json').exists():break
    time.sleep(2)
paths=[str((S/n).relative_to(R)) for n in ['export_start_evidence.py','.gitignore','START_EVIDENCE_RECEIPT.json']]
subprocess.run([m.core.GIT,'-C',str(R),'add','--',*paths],check=True)
if subprocess.run([m.core.GIT,'-C',str(R),'diff','--cached','--quiet']).returncode:subprocess.run([m.core.GIT,'-C',str(R),'commit','-m','audit: archive frozen crossscale setup numerical proofs profiles and actual checkpoint usage'],check=True,stdout=subprocess.DEVNULL)
print(json.dumps(receipt),flush=True)
