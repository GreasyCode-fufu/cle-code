"""从 train.py 抽取的归一化统计工具(仅数据统计, 不依赖任何模型文件)。

原 train.py 顶层 import 了 models.FasterNet 等无关主干, 会导致 CLE-Net 副本
被迫携带全部模型文件; 此处精确抽取 train_cle_fair.py 所需两个函数及其内部依赖。
逻辑与原实现逐行一致, 未做任何修改。
"""
import os
import time
import json
import hashlib
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _norm_transform_desc(img_size):
    resize_size = int(img_size * 1.143)
    return f"Resize({resize_size}) + CenterCrop({img_size}) + ToTensor()"

def _norm_cache_key(data_path, img_size):
    h = hashlib.md5(os.path.abspath(data_path).encode()).hexdigest()[:8]
    return f"{h}_img{img_size}"

def _dataset_fingerprint(image_paths, data_path):
    abs_root = os.path.abspath(data_path)
    parts = []
    for p in sorted(image_paths):
        rel = os.path.relpath(p, abs_root)
        sz = os.path.getsize(p)
        mt = os.path.getmtime(p)
        parts.append(f"{rel}|{sz}|{mt}")
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _load_norm_cache(cache_dir):
    path = os.path.join(cache_dir, _NORM_CACHE_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        warnings.warn(f"缓存文件 {path} 读取失败 ({e})，将重新计算 mean/std")
        return {}

def _save_norm_cache(cache_dir, cache):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, _NORM_CACHE_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def compute_dataset_mean_std(image_paths, img_size, batch_size=64, num_workers=4):
    if not image_paths:
        raise ValueError("image_paths 为空，无法计算 mean/std")

    resize_size = int(img_size * 1.143)
    temp_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    class _TempDataset(Dataset):
        def __init__(self, paths, tfm):
            self.paths = paths
            self.tfm = tfm
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            with Image.open(self.paths[idx]) as img:
                return self.tfm(img.convert("RGB"))

    loader = DataLoader(_TempDataset(image_paths, temp_transform),
                        batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=True)

    ch_sum = torch.zeros(3, dtype=torch.float64)
    ch_sum_sq = torch.zeros(3, dtype=torch.float64)
    n_pixels = 0
    for batch in loader:
        batch = batch.to(torch.float64)
        b, c, h, w = batch.shape
        n_pixels += b * h * w
        ch_sum += batch.sum(dim=[0, 2, 3])
        ch_sum_sq += (batch ** 2).sum(dim=[0, 2, 3])

    mean = ch_sum / n_pixels
    var = ch_sum_sq / n_pixels - mean ** 2
    var = torch.clamp(var, min=0.0)
    std = torch.clamp(torch.sqrt(var), min=1e-12)

    return mean.tolist(), std.tolist()

def get_or_compute_mean_std(data_path, image_paths, img_size,
                            cache_dir, batch_size=64, num_workers=4,
                            force_recompute=False):
    key = _norm_cache_key(data_path, img_size)
    transform_desc = _norm_transform_desc(img_size)
    num_images = len(image_paths)
    fingerprint = _dataset_fingerprint(image_paths, data_path)

    if not force_recompute:
        cache = _load_norm_cache(cache_dir)
        entry = cache.get(key)
        if entry is not None:
            valid = (
                entry.get("dataset_fingerprint") == fingerprint
                and entry.get("num_images") == num_images
                and entry.get("img_size") == img_size
                and entry.get("transform") == transform_desc
            )
            if valid:
                return entry["mean"], entry["std"], "hit"

    t0 = time.time()
    mean, std = compute_dataset_mean_std(image_paths, img_size, batch_size, num_workers)
    elapsed = time.time() - t0

    cache = _load_norm_cache(cache_dir)
    cache[key] = {
        "mean": mean,
        "std": std,
        "dataset_fingerprint": fingerprint,
        "num_images": num_images,
        "img_size": img_size,
        "resize_size": int(img_size * 1.143),
        "crop_size": img_size,
        "transform": transform_desc,
        "data_path": os.path.abspath(data_path),
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(elapsed, 1),
    }
    _save_norm_cache(cache_dir, cache)

    status = "forced recompute" if force_recompute else "miss"
    return mean, std, status
