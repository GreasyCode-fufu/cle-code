"""Single-image inference with a registered, SHA-verified val-selected checkpoint."""
from pathlib import Path
import argparse,json,sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import run_test_feedback_v2 as t
import torch
from PIL import Image
from torchvision import transforms as T

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',required=True);parser.add_argument('--seed',required=True,type=int,choices=[45,46,47])
    parser.add_argument('--image',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    assert args.method.replace('_','').isalnum()
    group=t.read(HERE/'methods_v2'/f'{args.method}.json');unit=next(u for u in group['units'] if u['seed']==args.seed)
    assert t.sha(t.ROOT/unit['checkpoint'])==unit['checkpoint_sha256']
    model=t.model_for(group,args.seed);state=torch.load(t.ROOT/unit['checkpoint'],map_location='cpu',weights_only=True)
    assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
    model.load_state_dict(state,strict=True);model.eval()
    resize,crop=(256,224) if group['recipe']=='A' else (448,384)
    with Image.open(args.image) as source:image=source.convert('RGB').resize((resize,resize),Image.Resampling.BILINEAR)
    x=T.Compose([T.CenterCrop(crop),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])(image)[None].cuda()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):logits=model(x)
    p=logits.float().softmax(1)[0].cpu().tolist()
    result=dict(method=args.method,seed=args.seed,checkpoint=unit['checkpoint'],checkpoint_sha256=unit['checkpoint_sha256'],
        prediction=int(max(range(3),key=lambda k:p[k])),probabilities={f'p{k}':p[k] for k in range(3)},
        resize=resize,crop=crop,note='Research model; class IDs retain the frozen dataset mapping. Single-image AMP may differ slightly from batch32 probabilities.')
    Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
