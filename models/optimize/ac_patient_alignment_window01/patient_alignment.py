"""Patient/class conditioned SupCon adaptation; no novelty or efficacy claim."""
import torch
from torch import nn
from torch.nn import functional as F

def pair_masks(labels,patients,mode):
    assert mode in ['all','cross','within'] and labels.ndim==patients.ndim==1 and labels.shape==patients.shape
    eye=torch.eye(len(labels),dtype=torch.bool,device=labels.device);same_label=labels[:,None]==labels[None,:];same_patient=patients[:,None]==patients[None,:]
    positive=same_label&~eye
    if mode=='cross':positive=positive&~same_patient
    if mode=='within':positive=positive&same_patient
    # A same-class pair excluded from positives is excluded from negatives too.
    # Different labels remain negatives, including mixed-label patients.
    allowed=(~same_label|positive)&~eye
    return positive,allowed

def contrastive_loss(z,labels,patients,mode='cross',temperature=.1):
    assert z.ndim==2 and len(z)==len(labels)==len(patients) and temperature>0
    dtype=torch.float64 if z.dtype==torch.float64 else torch.float32
    with torch.autocast(device_type=z.device.type,enabled=False):
        z=z.to(dtype);positive,allowed=pair_masks(labels,patients,mode);logits=z@z.T/temperature;logits=logits-logits.max(1,keepdim=True).values.detach()
        denominator=(logits.exp()*allowed).sum(1,keepdim=True).clamp_min(1e-12);log_probability=logits-denominator.log();count=positive.sum(1)
        per_anchor=-(positive*log_probability).sum(1)/count.clamp_min(1)
        # Average all anchors, as official SupCon; no-positive anchors contribute0.
        loss=per_anchor.mean();stats=dict(valid_anchor_fraction=float((count>0).float().mean().detach()),mean_positive_pairs=float(count.float().mean().detach()))
        return loss,stats

class PatientAlignmentNet(nn.Module):
    def __init__(self,backbone,kind,moment_class):
        super().__init__();assert kind in ['plain','moments','mom_cross','cross_only','mom_supcon','mom_within'];self.backbone=backbone;self.kind=kind;self.original_parameter_ids={id(p) for p in backbone.parameters()};self.features_for_loss=None
        devices=sorted({p.device.index for p in backbone.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            if kind not in ['plain','cross_only']:backbone.encoder[0][0][0]=moment_class(backbone.encoder[0][0][0],'moments')
            self.projector=None if kind in ['plain','moments'] else nn.Sequential(nn.Linear(768,128,bias=False),nn.GELU(),nn.Linear(128,64,bias=False)).to(next(backbone.parameters()).device)
    @property
    def encoder(self):return self.backbone.encoder
    @property
    def head(self):return self.backbone.head
    def forward(self,image):
        features=self.backbone.encoder(image);self.features_for_loss=features if self.training else None
        return self.backbone.head(features)
    def projected_features(self):
        assert self.training and self.projector is not None and self.features_for_loss is not None
        with torch.autocast(device_type=self.features_for_loss.device.type,enabled=False):
            value=self.features_for_loss.double() if self.features_for_loss.dtype==torch.float64 else self.features_for_loss.float()
            return F.normalize(self.projector(value),dim=1,eps=1e-12)
