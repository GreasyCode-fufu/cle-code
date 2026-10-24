"""Add geometric support while preserving the frozen RGB transform and RNG."""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from field_ops_v2 import unit_circle


class PairedFieldDataset:
 def __init__(self,base,*,resize,crop,training):
  self.base=base;self.indices=list(base.indices)
  self.template=Image.fromarray((unit_circle(resize,resize)[0,0].numpy()*255).astype(np.uint8))
  operations=([T.RandomHorizontalFlip(),T.RandomVerticalFlip(),T.RandomRotation(15),T.RandomCrop(crop)] if training else [T.CenterCrop(crop)])
  self.mask_transform=T.Compose(operations+[T.ToTensor()])
 def __len__(self):return len(self.base)
 def __getitem__(self,position):
  before=torch.get_rng_state()
  image,label,weight,index=self.base[position]
  after=torch.get_rng_state()
  try:
   torch.set_rng_state(before)
   support=self.mask_transform(self.template)
   assert torch.equal(torch.get_rng_state(),after),'paired transforms consumed different RNG draws'
  finally:torch.set_rng_state(after)
  assert image.shape[0]==3 and support.shape==(1,*image.shape[-2:])
  assert index==self.indices[position] and torch.isfinite(support).all()
  return torch.cat((image,support),dim=0),label,weight,index
