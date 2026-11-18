"""Use a completed formal field checkpoint with its paired geometric support."""
from pathlib import Path
import argparse,json,sys
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import run_field_structure as f
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from field_ops_v2 import unit_circle

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True,choices=f.ARMS);parser.add_argument('--seed',required=True,type=int,choices=f.SEEDS)
    parser.add_argument('--image',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    cfg=f.verify();job=next(j for j in f.READ(S/'jobs.json') if j['arm']==args.arm and j['seed']==args.seed)
    assert f.done(job),'Registered training and val prediction are not complete; no unfinished model is usable.'
    c=f.READ(S/'runs'/job['id']/'complete.json');assert f.sha(f.R/c['checkpoint'])==c['checkpoint_sha256']
    model=f.model_for(args.seed,args.arm);state=torch.load(f.R/c['checkpoint'],map_location='cpu',weights_only=True)
    assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point());model.load_state_dict(state,strict=True);model.eval()
    shape=f.q.RECIPES[cfg['recipe']];resize,crop=shape['resize'],shape['crop']
    with Image.open(args.image) as source:image=source.convert('RGB').resize((resize,resize),Image.Resampling.BILINEAR)
    rgb=T.Compose([T.CenterCrop(crop),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])(image)
    support=Image.fromarray((unit_circle(resize,resize)[0,0].numpy()*255).astype(np.uint8))
    mask=T.Compose([T.CenterCrop(crop),T.ToTensor()])(support);x=torch.cat((rgb,mask),0)[None].cuda()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):logits=model(x)
    p=logits.float().softmax(1)[0].cpu().tolist();result=dict(arm=args.arm,seed=args.seed,checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256'],
        family=cfg['family'],recipe=cfg['recipe'],prediction=int(np.argmax(p)),probabilities={f'p{k}':p[k] for k in range(3)},support='fixed geometric circle, no intensity or label inference',
        note='Research model; frozen class IDs. Single-image AMP can differ slightly from batch32 audit.')
    Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
