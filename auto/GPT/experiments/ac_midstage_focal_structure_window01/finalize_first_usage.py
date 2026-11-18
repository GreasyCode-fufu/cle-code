"""CPU-only actual checkpoint hash and first-CLI evidence export; no inference."""
from pathlib import Path
import hashlib,io,json,os,sys,tarfile,time,traceback
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));import run_module_structure as m
U=S/'usage_smoke'

def main():
 assert os.getuid()==1000 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 cfg=m.verify();m.SAVE(S/'usage_export_identity.json',dict(pid=os.getpid(),source_sha256=m.sha(__file__),started_at=m.core.timestamp(),CPU_only=True,new_training=0,new_inference=0))
 while not (U/'independent_cli_audit.json').exists():
  assert not (S/'usage_smoke_failure.json').exists() and not (S/'bootstrap_failure.json').exists()
  time.sleep(3)
 audit=m.READ(U/'independent_cli_audit.json');assert audit['passed'] and audit['index']==cfg['indices']['val'][0]==2386 and audit['max_probability_difference_batch1_vs_original_batch32']<1e-4
 job=next(j for j in m.READ(S/'jobs.json') if j['arm']=='mom_focalcontrast' and j['seed']==45)
 folder=S/'runs'/job['id'];comp=m.READ(folder/'complete.json');assert m.done(job) and comp['epochs_completed']==60
 checkpoint=m.R/comp['checkpoint'];assert checkpoint.resolve().is_relative_to(m.R.resolve()) and checkpoint.stat().st_uid==1000
 actual=m.sha(checkpoint);assert actual==comp['checkpoint_sha256']==audit['checkpoint_sha256']
 m.SAVE(U/'actual_weight_hash.json',dict(passed=True,actual_sha256=actual,checkpoint=comp['checkpoint'],checked_at=m.core.timestamp(),source_config_sha256=m.sha(S/'config.json'),hash_computed_on_server_actual_weight_bytes=True,new_training=0,new_inference=0))
 names={'independent_cli_audit.json':U/'independent_cli_audit.json','cli_prediction.json':U/'cli_prediction.json','actual_weight_hash.json':U/'actual_weight_hash.json',**{n:folder/n for n in ['complete.json','validation_complete.json','validation_predictions.csv','train_log.csv']}}
 data={n:p.read_bytes() for n,p in names.items()};manifest=dict(files=[dict(path=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in data.items()],actual_weight_sha256=actual,new_training=0,new_inference=0)
 out=U/'exports';out.mkdir(exist_ok=True);target=out/'midstage_focal_usage_inputs.tar.gz'
 with tarfile.open(target,'w:gz') as t:
  for n,b in data.items():
   item=tarfile.TarInfo(n);item.size=len(b);t.addfile(item,io.BytesIO(b))
  b=json.dumps(manifest,indent=2).encode();item=tarfile.TarInfo('EVIDENCE_MANIFEST.json');item.size=len(b);t.addfile(item,io.BytesIO(b))
 m.SAVE(out/'export_receipt.json',dict(passed=True,path=str(target.relative_to(m.R)),sha256=m.sha(target),bytes=target.stat().st_size,files=len(data),actual_weight_sha256=actual))
 print(json.dumps(m.READ(out/'export_receipt.json')),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:m.SAVE(S/'usage_export_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
