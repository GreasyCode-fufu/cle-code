"""Standalone frozen research models; checkpoint contains all encoder weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn
from PIL import Image
from torchvision import transforms as T

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class DinoClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        assert timm.__version__=='1.0.27'
        self.encoder=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=False,num_classes=0,img_size=224)
        self.head=nn.Linear(768,3)
    def forward(self,x):
        tokens=self.encoder.forward_features(x)
        return self.head(torch.cat((tokens[:,0],tokens[:,1:].mean(1)),dim=-1))


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        from models.Resnet import resnet34
        self.backbone=resnet34(num_classes=3)
    def forward(self,x):return self.backbone(x)


def preprocess(path):
    with Image.open(path) as image:
        image=image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
        return T.Compose([T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])(image)


def load_model(manifest,kind='dino',device='cuda'):
    cfg=json.loads(Path(manifest).read_text())
    selected=cfg['models'][kind];path=ROOT/selected['checkpoint']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==selected['checkpoint_sha256']
    model=(DinoClassifier() if kind=='dino' else CNNClassifier())
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    return model.to(device).eval().requires_grad_(False),cfg


def predict_paths(model,paths,device='cuda',batch_size=32):
    output=[]
    with torch.inference_mode():
        for start in range(0,len(paths),batch_size):
            batch=paths[start:start+batch_size]
            x=torch.stack([preprocess(p) for p in batch]).to(device)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=str(device).startswith('cuda')):
                logits=model(x)
            probabilities=logits.float().softmax(1).cpu().tolist()
            for path,p in zip(batch,probabilities):output.append(dict(path=str(path),pred=max(range(3),key=lambda k:p[k]),probabilities=p))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input');parser.add_argument('--kind',choices=['dino','cnn'],default='dino')
    parser.add_argument('--manifest',default=str(ROOT/'experiments/ac_final_evaluation/deployment_manifest.json'));parser.add_argument('--device',default='cuda')
    args=parser.parse_args();path=Path(args.input)
    paths=sorted(p for p in path.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}) if path.is_dir() else [path]
    assert paths
    torch.set_num_threads(1)
    model,cfg=load_model(args.manifest,args.kind,args.device)
    print(json.dumps(dict(class_names=cfg['class_names'],model=args.kind,results=predict_paths(model,paths,args.device)),ensure_ascii=False,indent=2))
