"""Single-image use of one registered seed and fixed four-angle probability mean."""
from pathlib import Path
import argparse,json,sys
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import run_orientation_reference as m
import torch
from PIL import Image
from torchvision import transforms as T

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',required=True,type=int,choices=[45,46,47]);parser.add_argument('--image',required=True);parser.add_argument('--output',required=True);a=parser.parse_args()
    cfg=m.verify();unit=next(u for u in cfg['units'] if u['seed']==a.seed);assert m.sha(m.R/unit['checkpoint'])==unit['checkpoint_sha256']
    model=m.q.model_for(a.seed,cfg['arm']);state=torch.load(m.R/unit['checkpoint'],map_location='cpu',weights_only=True)
    assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point());model.load_state_dict(state,strict=True);model.eval()
    shape=m.q.RECIPES[cfg['recipe']]
    with Image.open(a.image) as source:image=source.convert('RGB').resize((shape['resize'],shape['resize']),Image.Resampling.BILINEAR)
    x=T.Compose([T.CenterCrop(shape['crop']),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])(image)[None].cuda()
    views=[]
    with torch.inference_mode():
        for k in range(4):
            xx=x if k==0 else torch.rot90(x,k,(-2,-1)).contiguous()
            with torch.autocast('cuda',dtype=torch.float16):out=model(xx)
            views.append(out.float().softmax(1)[0].cpu().double())
    p=torch.stack(views).mean(0);result=dict(seed=a.seed,checkpoint=unit['checkpoint'],checkpoint_sha256=unit['checkpoint_sha256'],angles=cfg['angles'],aggregation=cfg['aggregation'],
        prediction=int(p.argmax()),probabilities={f'p{k}':float(p[k]) for k in range(3)},angle_probabilities={str(angle):z.tolist() for angle,z in zip(cfg['angles'],views)},
        note='One seed model, four fixed views. Study reports mean metrics over3 separately evaluated seeds, not a three-model ensemble. Single-image AMP may differ slightly from batch32 audit.')
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
