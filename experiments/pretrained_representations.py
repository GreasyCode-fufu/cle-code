"""DINOv2 official weights through the installed timm port; no new model claim."""
from pathlib import Path
import hashlib
import inspect
import torch
import timm
from timm.models import vision_transformer as vit
from timm.layers.pos_embed import resample_abs_pos_embed

DINO_SHA='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
DINO_URL='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def runtime():
    return dict(timm=timm.__version__,torch=torch.__version__,
                timm_vit_sha256=sha(inspect.getsourcefile(vit.VisionTransformer)),
                position_resampling_sha256=sha(inspect.getsourcefile(resample_abs_pos_embed)))

def dino(root):
    path=Path(root)/'weights/pretrained/dinov2_vits14_pretrain.pth'
    assert sha(path)==DINO_SHA and timm.__version__=='1.0.27'
    raw=torch.load(path,map_location='cpu',weights_only=True)
    model=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=False,num_classes=0,img_size=224)
    filtered=vit.checkpoint_filter_fn(dict(raw),model,antialias=False)
    assert set(raw)-set(filtered)=={'mask_token'}
    assert [k for k,v in filtered.items() if k in raw and not torch.equal(raw[k],v)]==['pos_embed']
    result=model.load_state_dict(filtered,strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert model.num_prefix_tokens==1 and model.embed_dim==384
    return model.eval().requires_grad_(False)

def state_sha(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

SCHEMES={'resnet_frozen_gap':[list(range(512))],
         'dino_cls':[list(range(512,896))],
         'dino_mean':[list(range(896,1280))],
         'dino_concat':[list(range(512,896)),list(range(896,1280))]}

def transform(feature,indices,kind):
    import numpy as np
    blocks=SCHEMES[kind];columns=np.array([i for block in blocks for i in block])
    raw=feature[np.asarray(indices)][:,columns].astype(np.float64)
    mean=raw.mean(0);scale=np.maximum(raw.std(0),1e-6);start=0
    for block in blocks:
        width=len(block);scale[start:start+width]*=np.sqrt(width*len(blocks));start+=width
    return (feature[:,columns].astype(np.float64)-mean)/scale,columns,mean,scale
