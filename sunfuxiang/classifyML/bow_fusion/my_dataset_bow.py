# -*- coding: utf-8 -*-
"""
my_dataset_bow.py — 支持 BoW 特征的自定义数据集 (classifyML/bow_fusion 版)

本文件改自 ../../BoW/my_dataset_bow_fixed.py，__getitem__ 返回 (img, bow_feat, label) 三元组，
供双流模型 (img, bow_feat) → logits 使用。classifyML 原始文件 (my_dataset.py) 未被修改。

两种 BoW 模式 (与 train_dual_fixed.py 的 --align-bow 对齐):
  align_bow=False (默认): 使用预提取的「全图」BoW 缓存 (行为与 BoW 版一致)。
  align_bow=True:         忽略缓存，在 __getitem__ 中一次性遍历 transform，
                          对「与 CNN 完全相同的裁剪区域」实时提取 LBP 直方图，
                          保证 BoW 特征与图像变换空间严格对齐。

相对 BoW 版的扩展:
  _aligned_transform 除支持 transforms.RandomResizedCrop 外，新增支持
  transforms.RandomCrop (train_cle_fair.py 的训练变换用的是 RandomCrop)。
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from bow_extractor import BOWFeatureCache, extract_lbp_from_pil


class MyDataSetBOW(Dataset):
    """带 BoW 特征的自定义数据集

    Args:
        images_path:  图像路径列表
        images_class: 图像标签列表
        transform:    图像变换 (仅 align_bow=False 时原样作用于 img)
        bow_cache:    BOWFeatureCache 实例 (align_bow=False 时使用)
        align_bow:    是否对齐 BoW 特征与图像裁剪区域
        scales:       LBP 多尺度参数 (仅 align_bow=True 时使用，默认 [16,32,64])
        K:            LBP 直方图 bin 数 (仅 align_bow=True 时使用，默认 256)
    """

    def __init__(self, images_path: list, images_class: list,
                 transform=None, bow_cache: BOWFeatureCache = None,
                 align_bow: bool = False,
                 scales: list = None, K: int = 256):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.bow_cache = bow_cache
        self.align_bow = align_bow
        self.scales = scales if scales is not None else [16, 32, 64]
        self.K = K

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.align_bow:
            # 一次性遍历 transform，保证 BoW 提取与图像裁剪严格对齐
            if self.transform is not None:
                img, bow_feat = self._aligned_transform(
                    img, self.transform, self.scales, self.K)
            else:
                bow_feat = extract_lbp_from_pil(img, self.scales, self.K)
        else:
            # 原有行为: 使用预提取的全图 BoW 缓存
            if self.transform is not None:
                img = self.transform(img)
            bow_feat = self.bow_cache[self.images_path[item]]

        return img, bow_feat, label

    @staticmethod
    def _aligned_transform(img, transform, scales, K):
        """一次遍历 transform，确保 BoW 提取与图像裁剪严格对齐。

        遇到 RandomResizedCrop / RandomCrop 时执行裁剪并记录裁剪结果 (PIL)，
        后续所有 transform 继续作用在同一裁剪图像上，
        最后对同一裁剪区域提取 BoW 特征，保证严格对齐。
        """
        pil_cropped = None
        img_out = img
        if isinstance(transform, transforms.Compose):
            for t in transform.transforms:
                if isinstance(t, transforms.RandomResizedCrop):
                    i, j, h, w = t.get_params(img_out, t.scale, t.ratio)
                    img_out = TF.resized_crop(
                        img_out, i, j, h, w, t.size, t.interpolation)
                    pil_cropped = img_out.convert('RGB') if img_out.mode != 'RGB' else img_out
                elif isinstance(t, transforms.RandomCrop):
                    i, j, h, w = t.get_params(img_out, t.size)
                    img_out = TF.crop(img_out, i, j, h, w)
                    pil_cropped = img_out.convert('RGB') if img_out.mode != 'RGB' else img_out
                else:
                    img_out = t(img_out)    # 其余 transform (翻转/旋转/ToTensor/Normalize 等) 正常执行
        else:
            img_out = transform(img_out)
        bow_src = pil_cropped if pil_cropped is not None else img
        bow_feat = extract_lbp_from_pil(bow_src, scales, K)
        return img_out, bow_feat

    @staticmethod
    def collate_fn(batch):
        images, bow_feats, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        bow_feats = torch.stack(bow_feats, dim=0)
        labels = torch.as_tensor(labels)
        return images, bow_feats, labels
