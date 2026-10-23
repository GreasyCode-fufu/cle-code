"""Rule out raw-vs-cache preprocessing differences at the original scale."""
from pathlib import Path
import sys
import torch
from PIL import Image
from torchvision import transforms as T
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import run_data_study as core
torch.set_num_threads(2)
cfg=core.read_json(core.STUDY/'config.json');records=core.read_json(core.STUDY/'manifest.json')
indices=cfg['val_indices'];cached=core.build_dataset(indices,False)
transform=T.Compose([T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
for pos,idx in enumerate(indices):
 with Image.open(ROOT/records[idx]['path']) as im:
  raw=transform(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
 old,label,weight,actual=cached[pos]
 assert idx==actual and label==records[idx]['label'] and torch.equal(raw,old)
out=ROOT/'experiments/scale_study/resolution_diagnostic/preprocessing_reproduction.json'
core.save_json(out,dict(passed=True,images=538,original_input_tensors_bitwise_equal=True,
                       protocol='Original raw PIL 256 resize/224 crop equals the frozen cache for every development image.',
                       implication='Scale diagnostic changed scale deliberately; raw-image preprocessing at 224 has no hidden discrepancy.'))
print('PASS all 538 original-scale input tensors match bitwise')
