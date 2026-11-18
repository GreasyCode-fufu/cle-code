# -*- coding: utf-8 -*-
"""
bow_extractor.py — 多尺度 LBP-BoW 特征提取与缓存 (classifyML/bow_fusion 版)

为 train_dual_fair.py 双流训练提供预提取的 LBP 直方图特征。
本文件改自 ../../BoW/bow_extractor.py，逻辑保持一致；classifyML 原始文件未被修改。

依赖: opencv-python (cv2), scikit-image (skimage), numpy, torch
"""

import os
import pickle
import numpy as np
import torch
import cv2
from skimage.feature import local_binary_pattern

SUPPORTED_EXTS = ['.jpg', '.JPG', '.png', '.PNG', '.jpeg']


def extract_and_cache_bow(image_dir, cache_path, scales=(16, 32, 64), K=256):
    """遍历图像目录，对每张图提取多尺度 LBP 直方图并拼接，缓存为 pkl 文件。

    Args:
        image_dir:  图像根目录 (含类别子文件夹)
        cache_path: 缓存 pkl 文件路径，存在则跳过
        scales:     多尺度半径基数列表 (默认 (16, 32, 64))
        K:          直方图 bin 数 (每尺度，默认 256)
    """
    if os.path.exists(cache_path):
        print(f"  BoW 缓存已存在，跳过提取: {cache_path}")
        return

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # 收集所有图像路径
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if os.path.splitext(f)[1] in SUPPORTED_EXTS:
                image_paths.append(os.path.join(root, f))

    print(f"  正在提取 {len(image_paths)} 张图像的 LBP-BoW 特征 ...")
    cache = {}
    for i, img_path in enumerate(image_paths):
        feat = _extract_lbp_histogram(img_path, scales, K)
        if feat is not None:
            cache[img_path] = feat
        if (i + 1) % 500 == 0 or (i + 1) == len(image_paths):
            print(f"    已处理 {i + 1}/{len(image_paths)}")

    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"  BoW 特征已缓存: {cache_path} ({len(cache)} 张图像)")


def _extract_lbp_histogram(img_path, scales, K):
    """对单张图像提取多尺度 LBP 直方图。

    每个尺度: P=8, R=scale//8, method='uniform', bins=K
    最终维度: len(scales) × K
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    histograms = []
    for scale in scales:
        R = scale // 8
        lbp = local_binary_pattern(img, P=8, R=R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=K, range=(0, K))
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        histograms.append(hist)

    return np.concatenate(histograms).astype(np.float32)


def extract_lbp_from_pil(pil_img, scales, K):
    """对 PIL 图像实时提取多尺度 LBP 直方图 (用于 align_bow 模式)。

    将 PIL Image 转灰度 → numpy → 逐尺度计算 LBP 直方图 → L1 归一化 → 拼接。
    返回 torch.Tensor，维度 = len(scales) × K。
    """
    img_gray = np.array(pil_img.convert('L'))
    histograms = []
    for scale in scales:
        R = scale // 8
        lbp = local_binary_pattern(img_gray, P=8, R=R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=K, range=(0, K))
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        histograms.append(hist)
    return torch.from_numpy(np.concatenate(histograms).astype(np.float32))


class BOWFeatureCache:
    """懒加载的 BoW 特征缓存。

    首次通过 __getitem__ 访问时才读取 pkl 文件。
    找不到对应图像时返回零向量。
    """

    def __init__(self, cache_path, bow_dim=768):
        self.cache_path = cache_path
        self.bow_dim = bow_dim
        self._cache = None

    def _load(self):
        if self._cache is None:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'rb') as f:
                    self._cache = pickle.load(f)
            else:
                self._cache = {}

    def __getitem__(self, image_path):
        self._load()
        feat = self._cache.get(image_path)
        if feat is not None:
            return torch.from_numpy(feat.copy())
        return torch.zeros(self.bow_dim, dtype=torch.float32)
