"""Validate the exact source/data/runtime and classifier initialization contract."""
import hashlib,json,os,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def check():
 lock=json.loads((ROOT/'protocol_lock.json').read_text())
 for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS'):
  if os.environ.get(key)!='2':raise RuntimeError(f'{key} must be 2; use ./run_reference.sh')
 import torch,torchvision,numpy,PIL
 actual=dict(python='.'.join(map(str,sys.version_info[:3])),torch=torch.__version__,
             torchvision=torchvision.__version__,numpy=numpy.__version__,pillow=PIL.__version__)
 if actual!=lock['versions']:raise RuntimeError(f'Runtime mismatch: expected {lock["versions"]}, actual {actual}')
 for relative,digest in json.loads((ROOT/'source_files.sha256.json').read_text()).items():
  if sha(ROOT/relative)!=digest:raise RuntimeError('Historical source modified: '+relative)
 if sha(lock['pretrained'])!=lock['pretrained_sha256']:raise RuntimeError('Pretrained checkpoint mismatch')
 manifest=ROOT/'environment/dataset_manifest.json'
 if sha(manifest)!=lock['dataset_manifest_sha256']:raise RuntimeError('Dataset manifest modified')
 entries=json.loads(manifest.read_text());data=Path(lock['dataset'])
 expected=set()
 for e in entries:
  name=(e['split'],e['path']);expected.add(name)
  if sha(data/e['split']/e['path'])!=e['sha256']:raise RuntimeError('Dataset bytes changed: '+str(name))
 actual_files=set()
 for split in ('train','val','test'):
  for p in (data/split).rglob('*'):
   if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff'):
    actual_files.add((split,str(p.relative_to(data/split))))
 if actual_files!=expected:raise RuntimeError('Dataset file set changed')
 sys.path.insert(0,str(ROOT))
 import train_cle_fair
 train_cle_fair.set_seed(42)
 model=train_cle_fair.build_model('convnext_egb_scg',num_classes=3)
 h=hashlib.sha256(model.head.weight.detach().contiguous().numpy().tobytes()).hexdigest()
 if h!=lock['head_weight_seed42_sha256']:raise RuntimeError('Classifier initialization mismatch: '+h)
 return dict(status='PASS',versions=actual,head_weight_sha256=h,source_commit=lock['historical_commit'],
             images=len(entries),pretrained_sha256=lock['pretrained_sha256'])

if __name__=='__main__':print(json.dumps(check(),ensure_ascii=False,indent=2))
