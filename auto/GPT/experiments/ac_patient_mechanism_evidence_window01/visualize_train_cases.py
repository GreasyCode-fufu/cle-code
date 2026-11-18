"""Fixed train-only qualitative view; never used to filter or relabel experiments."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageOps,ImageDraw
S=Path(__file__).resolve().parent;R=S.parents[1]
records=json.loads((S/'manifest.json').read_text());rng=np.random.default_rng(20261024);chosen=[];used=set()
for label in range(3):
    rows=[row for row in records if row['split']=='train' and row['label']==label];count=0
    for pos in rng.permutation(len(rows)):
        row=rows[int(pos)]
        if row['group_id'] in used:continue
        chosen.append(row);used.add(row['group_id']);count+=1
        if count==4:break
assert len(chosen)==len(used)==12
canvas=Image.new('RGB',(1024,858),(25,25,25));draw=ImageDraw.Draw(canvas)
for pos,row in enumerate(chosen):
    source=R/row['path'];assert hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256']
    with Image.open(source) as raw:image=ImageOps.contain(raw.convert('RGB'),(256,256),Image.Resampling.BILINEAR)
    x=pos%4*256;y=pos//4*286;canvas.paste(image,(x+(256-image.width)//2,y+(256-image.height)//2))
    draw.text((x+4,y+259),f'class{row["label"]} idx{row["index"]} grade{row["grade"]}',fill='white')
folder=S/'visualization';folder.mkdir(exist_ok=True);canvas.save(folder/'train_contact_sheet.png')
(folder/'manifest.json').write_text(json.dumps(dict(split='train',selection='seed20261024;4 distinct patients per class,12 total; visualization only, no data filtering or relabeling',rows=chosen),indent=2),encoding='utf-8')
