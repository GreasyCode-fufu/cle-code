"""Use an actually completed val-selected module checkpoint; never silently use random weights."""
from pathlib import Path
import argparse,json,sys
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import run_module_structure as m
import torch
from PIL import Image
from torchvision.transforms import functional as tf

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=m.ARMS);p.add_argument('--seed',required=True,type=int,choices=m.SEEDS);p.add_argument('--image',required=True,type=Path);p.add_argument('--views',choices=['single','c4'],default='single');p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    cfg=m.verify();job=next(j for j in m.READ(S/'jobs.json') if j['arm']==a.arm and j['seed']==a.seed)
    assert m.done(job),'This method/seed has no complete training and val-selected checkpoint yet; no prediction produced'
    c=m.READ(S/'runs'/job['id']/'complete.json');assert m.sha(m.R/c['checkpoint'])==c['checkpoint_sha256']
    model=m.model_for(a.seed,a.arm);model.load_state_dict(torch.load(m.R/c['checkpoint'],map_location='cpu',weights_only=True),strict=True);model.eval()
    assert a.image.is_absolute() and a.image.is_file()
    with Image.open(a.image) as image:image=image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
    x=tf.normalize(tf.to_tensor(tf.center_crop(image,[224,224])),[.485,.456,.406],[.229,.224,.225])[None].cuda();views=[]
    with torch.inference_mode():
        for k in ([0] if a.views=='single' else [0,1,2,3]):
            with torch.autocast('cuda',dtype=torch.float16):logits=model(torch.rot90(x,k,(-2,-1)).contiguous())
            views.append(logits.float().softmax(1).cpu())
    probs=torch.stack(views).mean(0)[0];assert torch.isfinite(probs).all()
    result=dict(arm=a.arm,seed=a.seed,checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256'],source_config_sha256=m.sha(S/'config.json'),image_sha256=m.sha(a.image),views=a.views,predicted_class=int(probs.argmax()),probabilities=probs.tolist(),class_ids=[0,1,2],malignant_class_id=1,scope='Research model prediction; no independent clinical validation claimed')
    a.output.parent.mkdir(parents=True,exist_ok=True);m.SAVE(a.output,result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
