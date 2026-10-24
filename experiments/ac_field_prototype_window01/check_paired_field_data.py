"""Full allowed-cohort tensor/identity/RNG audit; no neural model or test."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
S=Path(__file__).resolve().parent;R=S.parents[1];Q=R/'experiments/ac_qualification_20261024_window01'
sys.path[:0]=[str(S),str(Q)]
import run_qualification as q
from paired_field_data import PairedFieldDataset
from field_ops_v2 import unit_circle
torch.set_num_threads(1)
cfg=json.loads((Q/'config.json').read_text());rows=[]
for recipe,shape in [('A',(256,224)),('B',(448,384))]:
 resize,crop=shape
 for split in ['train','val']:
  indices=cfg['indices'][split];training=split=='train'
  base=q.GuardedData(indices,training,recipe);paired=PairedFieldDataset(base,resize=resize,crop=crop,training=training)
  seen=[];maximum=0.
  torch.manual_seed(20261024)
  for position in range(len(base)):
   before=torch.get_rng_state();expected=base[position];after=torch.get_rng_state()
   torch.set_rng_state(before);actual=paired[position];actual_after=torch.get_rng_state()
   assert torch.equal(expected[0],actual[0][:3]) and expected[1:]==actual[1:]
   assert torch.equal(after,actual_after) and len(actual[0])==4
   maximum=max(maximum,float((expected[0]-actual[0][:3]).abs().max()));seen.append(actual[3])
  assert seen==indices and len(seen)==len(set(seen))
  rows.append(dict(recipe=recipe,split=split,n=len(seen),rgb_max_difference=maximum,all_rng_states_exact=True,
   label_weight_index_exact=True,index_sha256=hashlib.sha256(json.dumps(seen).encode()).hexdigest()))
 # Independent alpha-channel oracle for the replayed mask, using source image/mask RGBA as one geometric object.
 recordmap={r['index']:r for r in json.loads((R/'experiments/ac_dataset_v1/dataset_manifest.json').read_text())}
 transforms=T.Compose([T.RandomHorizontalFlip(),T.RandomVerticalFlip(),T.RandomRotation(15),T.RandomCrop(crop),T.ToTensor()])
 base=q.GuardedData(cfg['indices']['train'][:4],True,recipe);paired=PairedFieldDataset(base,resize=resize,crop=crop,training=True)
 for position,index in enumerate(base.indices):
  source=R/recordmap[index]['path']
  with Image.open(source) as im:rgba=im.convert('RGB').resize((resize,resize),Image.Resampling.BILINEAR)
  rgba.putalpha(Image.fromarray((unit_circle(resize,resize)[0,0].numpy()*255).astype(np.uint8)))
  for seed in range(20):
   torch.manual_seed(seed);actual=paired[position]
   torch.manual_seed(seed);oracle=transforms(rgba)
   assert torch.equal(actual[0][3:],oracle[3:])
 rows.append(dict(recipe=recipe,alpha_geometry_oracle_cases=80,all_masks_exact=True))
result=dict(passed=True,scope='All frozen AC train2077/val427 for both recipes; no neural fitting or test access',
 checks=rows,full_allowed_tensor_comparisons=5008,rgba_geometry_oracles=160,
 source_sha256={n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in ['paired_field_data.py','check_paired_field_data.py','PAIRED_FIELD_DATA_PROTOCOL.md']},
 upstream_dataset_source_sha256=hashlib.sha256((Q/'run_qualification.py').read_bytes()).hexdigest(),
 qualification_config_sha256=hashlib.sha256((Q/'config.json').read_bytes()).hexdigest())
(S/'paired_dataset_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(passed=True,tensor_comparisons=5008,geometry_oracles=160,test_images=0,cle_trainings=0)))
