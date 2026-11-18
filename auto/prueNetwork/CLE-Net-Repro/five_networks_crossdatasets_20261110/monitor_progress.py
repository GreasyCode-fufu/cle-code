import csv,json
from pathlib import Path
root=Path('/home/login/data/cle/auto/prueNetwork/CLE-Net-Repro/five_networks_crossdatasets_20261110')
s=json.loads((root/'status.json').read_text())
out={k:s.get(k) for k in ['time','phase','dataset','model','seed','stage','completed','total'] if k in s}
if (root/'per_seed.csv').exists():out['completed_tests']=len(list(csv.DictReader((root/'per_seed.csv').open(encoding='utf-8-sig'))))
if 'model' in s and 'dataset' in s and 'seed' in s:
 f=root/f"runs/{s['dataset']}/{s['model']}_seed{s['seed']}/{s['model']}/train_log.csv"
 if f.exists():
  rows=list(csv.DictReader(f.open()))
  if rows:out['last_epoch']=rows[-1]
if s.get('traceback'):out['error']=s['traceback']
print(json.dumps(out))
