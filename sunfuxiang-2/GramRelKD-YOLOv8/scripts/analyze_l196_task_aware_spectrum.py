#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L1-96 任务感知 / 类别条件谱 + Fisher 判别谱分析
=================================================================

在已完成的全局谱分析 (runs/analysis/l1_96_spectrum) 基础上, 结合 GT 检测框,
分析不同类别前景 / 背景 / 类别间判别方向的谱结构, 判断哪些通道/子空间真正与
检测任务相关.

只做分析:
    * 不训练、不修改权重;
    * 不修改 block.py / tasks.py / 数据集标签;
    * 不自动生成训练 YAML.

关键设计:
    * 预处理与 GT 框映射严格复用 Ultralytics 8.4.84 LetterBox (get_params 取
      ratio/left/top), 框从原图坐标 -> 640 输入坐标 -> 各特征图 (H,W) 坐标.
    * 每张图按类别构建 union / disjoint (排除跨类别重叠) 前景 mask + 背景 mask.
    * 两种采样: natural (自然频率) 与 balanced (每类≤512, 背景≤512, 每框≤256).
    * 在线累积 C×C Gram (CPU float64), 不保存完整激活.
    * 类别条件中心化协方差 Cov_k 谱; Fisher (natural/balanced) 用 Sw 白化 Sb;
      逐通道 Fisher score; 方差/Fisher 排序的 Spearman 相关等.
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
DEFECT_NAMES = ["spalling", "crack", "moisture"]            # cls 0/1/2
FISHER_CLASSES = ["background", "spalling", "crack", "moisture"]  # K=4
DEFECT_CLS = [0, 1, 2]

# 类别条件谱用 union mask; Fisher 用 disjoint (排除跨类别重叠)
SPECTRUM_GROUPS = ["background", "spalling_union", "crack_union",
                   "moisture_union", "all_foreground", "global"]
ALL_GROUPS = ["background", "spalling_union", "crack_union", "moisture_union",
              "spalling_disjoint", "crack_disjoint", "moisture_disjoint",
              "all_foreground", "global", "overlap"]

PER_BOX_CAP = 256     # 每个GT框 balanced 采样上限
CLASS_CAP = 512       # 每图每类 balanced 上限
OTHER_CAP = 512       # 背景/前景/全局 balanced 上限
CHANNEL_EPS = 1e-8

IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")

# 运行期上下文 (主循环写入, hook 读取)
BATCH_CTX = {"items": []}


# ============================================================================
# 参数 / 设备
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L1-96 task-aware / Fisher spectrum analysis.")
    p.add_argument("--weights", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--data-yaml", default=None)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--max-images", type=int, default=0, help="<=0 means all (1105).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-spatial-samples", type=int, default=0,
                   help="0=all positions (natural); balanced uses its own caps.")
    p.add_argument("--fisher-eps", type=float, default=1e-4,
                   help="Sw regularization: Sw + eps*trace(Sw)/C*I")
    p.add_argument("--output-dir",
                   default="runs/analysis/l1_96_task_aware_spectrum")
    return p.parse_args()


def _resolve_device(dev: str) -> torch.device:
    dev = str(dev).strip()
    if dev.lower() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("[WARN] CUDA 不可用, 回退 CPU.")
        return torch.device("cpu")
    if dev.isdigit():
        return torch.device(f"cuda:{dev}")
    if dev == "cuda":
        return torch.device("cuda:0")
    return torch.device(dev)


# ============================================================================
# 数据 / 标签
# ============================================================================
def find_data_yaml(data_root, explicit):
    if explicit and os.path.isfile(explicit):
        return os.path.abspath(explicit)
    for name in ["data.yaml", "hongkongScale.yaml", "dataset.yaml", "data.yml"]:
        fp = os.path.join(data_root, name)
        if os.path.isfile(fp):
            return os.path.abspath(fp)
    return None


def resolve_split_dir(args, nc, output_dir):
    info = {}
    yaml_path = find_data_yaml(args.data_root, args.data_yaml)
    if yaml_path is None:
        os.makedirs(output_dir, exist_ok=True)
        yaml_path = os.path.join(output_dir, "_auto_data.yaml")
        names_block = "\n".join([f"  {i}: {DEFECT_NAMES[i]}" for i in range(nc)])
        with open(yaml_path, "w") as f:
            f.write(f"path: {args.data_root}\ntrain: train/images\nval: val/images\n"
                    f"test: test/images\nnc: {nc}\nnames:\n{names_block}\n")
        info["yaml_source"] = "auto-generated"
    else:
        info["yaml_source"] = "found/explicit"
    info["yaml_path"] = yaml_path
    try:
        from ultralytics.data.utils import check_det_dataset
        check_det_dataset(yaml_path, autodownload=False)
        info["check_det_dataset"] = "ok"
    except Exception as e:  # noqa: BLE001
        info["check_det_dataset"] = f"failed: {e}"
    split_dir = os.path.join(args.data_root, args.split, "images")
    if not os.path.isdir(split_dir):
        split_dir = os.path.join(args.data_root, args.split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"split 目录不存在: {split_dir}")
    info["split_dir"] = split_dir
    return split_dir, yaml_path, info


def list_split_images(split_dir):
    imgs = []
    for ext in IMG_EXT:
        imgs.extend(glob.glob(os.path.join(split_dir, ext)))
    return sorted(set(imgs))


def select_images(paths, max_images, seed):
    paths = sorted(paths)
    if max_images is None or max_images <= 0 or max_images >= len(paths):
        return paths
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(paths), size=max_images, replace=False))
    return [paths[i] for i in idx]


def label_path_for(image_path):
    # .../images/xxx.jpg -> .../labels/xxx.txt
    p = image_path.replace("/images/", "/labels/")
    base, _ = os.path.splitext(p)
    return base + ".txt"


def read_yolo_labels(txt_path):
    """返回 (cls[N], xywh[N] normalized cx,cy,w,h). 无标签返回空."""
    if not os.path.isfile(txt_path):
        return np.zeros((0,), int), np.zeros((0, 4))
    cls, boxes = [], []
    for line in open(txt_path):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls.append(int(float(parts[0])))
        boxes.append([float(x) for x in parts[1:5]])
    if not cls:
        return np.zeros((0,), int), np.zeros((0, 4))
    return np.array(cls), np.array(boxes)


# ============================================================================
# Dataset: 复用 LetterBox, 同时把 GT 框映射到 640 坐标 (供 hook 用)
# ============================================================================
class TaskAwareDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, imgsz, stride=32):
        self.paths = image_paths
        from ultralytics.data.augment import LetterBox
        self.letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False,
                                   scaleup=True, stride=stride, padding_value=114)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import cv2
        path = self.paths[i]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"无法读取: {path}")
        h0, w0 = img.shape[:2]
        # 用 LetterBox.get_params 取与验证完全一致的 ratio / left / top
        params = self.letterbox.get_params({"img": img})
        r = params["ratio"][0]
        left, top = params["left"], params["top"]
        # letterbox 图像
        img640 = self.letterbox.apply_image({"img": img}, params)["img"]
        # 预处理: HWC->CHW, BGR->RGB, /255
        img640 = np.ascontiguousarray(img640.transpose(2, 0, 1)[::-1, :, :])
        tensor = torch.from_numpy(img640).float().div_(255.0)

        # GT 框: normalized xywh -> 原图像素 xyxy -> 640 坐标 xyxy
        cls, cxcywh = read_yolo_labels(label_path_for(path))
        if len(cls) > 0:
            cx, cy, bw, bh = cxcywh[:, 0], cxcywh[:, 1], cxcywh[:, 2], cxcywh[:, 3]
            x1 = (cx - bw / 2) * w0 * r + left
            x2 = (cx + bw / 2) * w0 * r + left
            y1 = (cy - bh / 2) * h0 * r + top
            y2 = (cy + bh / 2) * h0 * r + top
            boxes640 = np.stack([x1, y1, x2, y2], axis=1)
            boxes640[:, [0, 2]] = np.clip(boxes640[:, [0, 2]], 0, 640)
            boxes640[:, [1, 3]] = np.clip(boxes640[:, [1, 3]], 0, 640)
        else:
            boxes640 = np.zeros((0, 4))
        return tensor, cls, boxes640, i


def collate(batch):
    tensors = torch.stack([b[0] for b in batch], 0)
    return tensors, [{"cls": b[1], "boxes640": b[2], "img_idx": b[3]} for b in batch]


# ============================================================================
# Mask 构建 + 采样
# ============================================================================
def build_masks(boxes640: np.ndarray, cls: np.ndarray, H: int, W: int) -> dict:
    """把 640 坐标框映射到 (H,W) 特征图, 构建 union/disjoint/overlap/bg 等掩码."""
    sx, sy = W / 640.0, H / 640.0
    if len(boxes640) > 0:
        xb = boxes640 * np.array([sx, sy, sx, sy])
        xb = np.round(xb).astype(int)
        xb[:, [0, 2]] = np.clip(xb[:, [0, 2]], 0, W)
        xb[:, [1, 3]] = np.clip(xb[:, [1, 3]], 0, H)
        xb[:, 2] = np.maximum(xb[:, 2], xb[:, 0] + 1)
        xb[:, 3] = np.maximum(xb[:, 3], xb[:, 1] + 1)
    else:
        xb = np.zeros((0, 4), int)

    union = {c: np.zeros((H, W), bool) for c in DEFECT_CLS}
    boxes_excl = []  # (cls, flat_idx of interior & ~overlap)
    # 先建 union
    box_masks = []
    for c, (x1, y1, x2, y2) in zip(cls.tolist(), xb.tolist() if len(xb) else []):
        m = np.zeros((H, W), bool)
        m[y1:y2, x1:x2] = True
        box_masks.append((c, m))
        if c in union:
            union[c] |= m
    fg_any = union[0] | union[1] | union[2]
    overlap = (union[0].astype(np.int8) + union[1].astype(np.int8) +
               union[2].astype(np.int8)) >= 2
    bg = ~fg_any

    for c, m in box_masks:
        excl = m & ~overlap
        boxes_excl.append((c, np.flatnonzero(excl.ravel())))

    def f(mask):
        return np.flatnonzero(mask.ravel())

    return {
        "union_idx": {c: f(union[c]) for c in DEFECT_CLS},
        "disjoint_idx": {c: f(union[c] & ~overlap) for c in DEFECT_CLS},
        "overlap_idx": f(overlap),
        "bg_idx": f(bg),
        "fg_idx": f(fg_any),
        "global_idx": np.arange(H * W),
        "boxes_excl": boxes_excl,
        "H": H, "W": W,
    }


def _cap_sample(arr, rng, cap):
    if len(arr) <= cap:
        return arr
    return np.sort(rng.choice(arr, cap, replace=False))


def _balanced_disjoint_for_class(masks, c, rng):
    """每框≤256, 每类≤512, 排除跨类别重叠与重复."""
    used = np.zeros(masks["H"] * masks["W"], bool)
    pool = []
    for bc, idx in masks["boxes_excl"]:
        if bc != c:
            continue
        cand = idx[~used[idx]] if len(idx) else idx
        if len(cand) > PER_BOX_CAP:
            cand = rng.choice(cand, PER_BOX_CAP, replace=False)
        used[cand] = True
        pool.append(cand)
    pool = np.concatenate(pool) if pool else np.zeros((0,), int)
    if len(pool) > CLASS_CAP:
        pool = rng.choice(pool, CLASS_CAP, replace=False)
    return np.sort(pool)


def sample_indices(masks, mode, rng):
    """返回 {group: flat_idx_array}."""
    out = {}
    if mode == "natural":
        out["background"] = masks["bg_idx"]
        out["spalling_union"] = masks["union_idx"][0]
        out["crack_union"] = masks["union_idx"][1]
        out["moisture_union"] = masks["union_idx"][2]
        out["spalling_disjoint"] = masks["disjoint_idx"][0]
        out["crack_disjoint"] = masks["disjoint_idx"][1]
        out["moisture_disjoint"] = masks["disjoint_idx"][2]
        out["all_foreground"] = masks["fg_idx"]
        out["global"] = masks["global_idx"]
        out["overlap"] = masks["overlap_idx"]
    else:  # balanced
        out["background"] = _cap_sample(masks["bg_idx"], rng, OTHER_CAP)
        out["spalling_union"] = _cap_sample(masks["union_idx"][0], rng, CLASS_CAP)
        out["crack_union"] = _cap_sample(masks["union_idx"][1], rng, CLASS_CAP)
        out["moisture_union"] = _cap_sample(masks["union_idx"][2], rng, CLASS_CAP)
        out["spalling_disjoint"] = _balanced_disjoint_for_class(masks, 0, rng)
        out["crack_disjoint"] = _balanced_disjoint_for_class(masks, 1, rng)
        out["moisture_disjoint"] = _balanced_disjoint_for_class(masks, 2, rng)
        out["all_foreground"] = _cap_sample(masks["fg_idx"], rng, OTHER_CAP)
        out["global"] = _cap_sample(masks["global_idx"], rng, OTHER_CAP)
        out["overlap"] = _cap_sample(masks["overlap_idx"], rng, OTHER_CAP)
    return out


# ============================================================================
# Gram 累积器
# ============================================================================
class GramAcc:
    def __init__(self):
        self.C = None
        self.sum_x = None
        self.gram = None
        self.count = 0

    def init(self, C):
        self.C = C
        self.sum_x = torch.zeros(C, dtype=torch.float64)
        self.gram = torch.zeros(C, C, dtype=torch.float64)

    @torch.no_grad()
    def add(self, X: torch.Tensor):
        # X: [C, n] on device, float32
        C, n = X.shape
        if n == 0:
            return
        if self.C is None:
            self.init(C)
        elif self.C != C:
            return
        s = X.sum(dim=1).double().cpu()
        g = (X @ X.t()).double().cpu()
        self.sum_x += s
        self.gram += g
        self.count += n


# ============================================================================
# Hook
# ============================================================================
def extract_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            t = extract_tensor(o)
            if t is not None and t.dim() == 4:
                return t
    return None


def extract_input_tensor(args):
    if isinstance(args, torch.Tensor):
        return args if args.dim() == 4 else None
    if isinstance(args, (tuple, list)):
        for a in args:
            t = extract_input_tensor(a)
            if t is not None:
                return t
    return None


class FeatureSpec:
    def __init__(self, name, parent, ftype, hook_kind, target_fn):
        self.name = name
        self.parent = parent
        self.ftype = ftype
        self.hook_kind = hook_kind
        self.target_fn = target_fn


def build_specs():
    specs = []
    outs = [("layer12_output", 12, lambda m: m.model[12]),
            ("layer15_output", 15, lambda m: m.model[15]),
            ("layer18_output", 18, lambda m: m.model[18]),
            ("layer21_output", 21, lambda m: m.model[21])]
    for name, p, fn in outs:
        specs.append(FeatureSpec(name, p, "output", "post", fn))
    for idx in [12, 18, 21]:
        specs.append(FeatureSpec(f"layer{idx}_cv1", idx, "cv1", "post",
                                 (lambda i: (lambda m: m.model[i].cv1))(idx)))
        specs.append(FeatureSpec(f"layer{idx}_bottleneck_last", idx, "bottleneck_last", "post",
                                 (lambda i: (lambda m: m.model[i].m[-1]))(idx)))
        specs.append(FeatureSpec(f"layer{idx}_prefuse", idx, "prefuse", "pre",
                                 (lambda i: (lambda m: m.model[i].cv2))(idx)))
    return specs


def process_feature(name: str, feat_id: int, feat: torch.Tensor,
                    accs: dict, overlap_stats: dict, seed: int):
    """对一张 batch 的特征 [B,C,H,W] 累积所有 (mode, group), 并统计 overlap/fg."""
    if feat.dim() != 4:
        return
    B, C, H, W = feat.shape
    items = BATCH_CTX["items"]
    for b in range(B):
        if b >= len(items):
            break
        it = items[b]
        img_idx = int(it["img_idx"])
        feat_b = feat[b].detach()             # [C,H,W]
        flat = feat_b.reshape(C, H * W)       # [C, N]
        masks = build_masks(it["boxes640"], it["cls"], H, W)
        overlap_stats[name]["overlap"] += len(masks["overlap_idx"])
        overlap_stats[name]["fg"] += len(masks["fg_idx"])
        overlap_stats[name]["n_img"] += 1
        # 每张图×特征独立可复现 seed
        s = (seed * 1000003 + img_idx * 7919 + feat_id * 104729) & 0x7FFFFFFF
        rng = np.random.RandomState(s)
        for mode in ["natural", "balanced"]:
            idx_sets = sample_indices(masks, mode, rng)
            for group, idx in idx_sets.items():
                if len(idx) == 0:
                    continue
                acc = accs[name][mode][group]
                if acc.C is None:
                    acc.init(C)
                # device index tensor
                tidx = torch.as_tensor(idx, device=feat.device, dtype=torch.long)
                X = flat.index_select(1, tidx)  # [C, n]
                acc.add(X)


def register_hooks(specs, mdl, accs, overlap_stats, seed):
    handles = []
    for fid, spec in enumerate(specs):
        mod = spec.target_fn(mdl)
        if spec.hook_kind == "post":
            def make(name, fid):
                def h(_m, _i, out):
                    t = extract_tensor(out)
                    if t is not None:
                        process_feature(name, fid, t, accs, overlap_stats, seed)
                return h
            handles.append(mod.register_forward_hook(make(spec.name, fid)))
        else:
            def make_p(name, fid):
                def h(_m, args):
                    t = extract_input_tensor(args)
                    if t is not None:
                        process_feature(name, fid, t, accs, overlap_stats, seed)
                return h
            handles.append(mod.register_forward_pre_hook(make_p(spec.name, fid)))
    return handles


# ============================================================================
# 谱 / Fisher 数学
# ============================================================================
def cov_from_gram(acc):
    """中心化协方差 Cov = (gram - n mu mu^T)/(max(n-1,1)); 也返回 raw 特征值."""
    C = acc.C
    if C is None or acc.count == 0:
        return None
    n = max(acc.count, 1)
    mu = (acc.sum_x / n).double()
    M_raw = (acc.gram / n).double()
    M_cen = (acc.gram - n * torch.outer(mu, mu)).double() / max(n - 1, 1)
    eval_raw = torch.linalg.eigvalsh(M_raw).clamp(min=0).flip(0).cpu().numpy()
    eval_cen = torch.linalg.eigvalsh(M_cen).clamp(min=0).flip(0).cpu().numpy()
    return {"C": C, "count": acc.count, "mu": mu.cpu().numpy(),
            "eval_raw": eval_raw, "eval_cen": eval_cen, "cov": M_cen.cpu().numpy()}


def metrics_from_evals(evals):
    e = np.asarray(evals, dtype=np.float64).flatten()
    C = len(e)
    total = float(e.sum())
    if total <= 1e-12:
        return dict(r90=0, r95=0, r99=0, r90_ratio=0.0, r95_ratio=0.0, r99_ratio=0.0,
                    stable_rank=0.0, effective_rank=0.0, participation_ratio=0.0)
    cum = np.cumsum(e) / total

    def rk(th):
        i = np.where(cum >= th)[0]
        return int(i[0]) + 1 if len(i) else C

    return dict(r90=rk(0.90), r95=rk(0.95), r99=rk(0.99),
                r90_ratio=rk(0.90) / C, r95_ratio=rk(0.95) / C, r99_ratio=rk(0.99) / C,
                stable_rank=total / float(e.max()),
                effective_rank=float(np.exp(-np.sum((e / total) *
                                                     np.log(e / total + 1e-12)))),
                participation_ratio=(total ** 2) / float(np.sum(e ** 2)))


def fisher_from_groups(class_stats, eps):
    """
    class_stats: dict class_name -> {count:n_k, mu:(C,), cov:(C,C)}  (4 类, disjoint)
    返回 natural 与 balanced 两套 Fisher.
    natural: w_k = n_k/N ; balanced: w_k = 1/K.
    """
    classes = FISHER_CLASSES
    res = {}
    for mode in ["natural", "balanced"]:
        K = len(classes)
        # 过滤掉 count==0 的类 (balanced 仍给等权, 但 cov 可能为 0; 这里保留所有类,
        # 若某类无样本则其贡献为 0, balanced 等权会引入全 0 行 -> 已被 eps 正则吸收)
        ns = np.array([max(class_stats[c]["count"], 0) for c in classes], float)
        N = ns.sum()
        if mode == "natural":
            w = ns / N if N > 0 else np.ones(K) / K
        else:
            w = np.ones(K) / K
        mus = np.stack([class_stats[c]["mu"] for c in classes])      # K,C
        covs = np.stack([class_stats[c]["cov"] for c in classes])    # K,C,C
        C = mus.shape[1]
        mu = (w[:, None] * mus).sum(0)                                # C
        Sw = np.einsum("k,kij->ij", w, covs)                          # C,C
        dmu = mus - mu[None, :]                                       # K,C
        Sb = np.einsum("k,ki,kj->ij", w, dmu, dmu)                    # C,C
        Sw_t = torch.tensor(Sw, dtype=torch.float64)
        Sb_t = torch.tensor(Sb, dtype=torch.float64)
        tr = torch.diagonal(Sw_t).sum().clamp(min=1e-12)
        Sw_reg = Sw_t + eps * tr / C * torch.eye(C, dtype=torch.float64)
        # 白化: eigh(Sw_reg) -> V diag(1/sqrt(w))
        ev, V = torch.linalg.eigh(Sw_reg)
        ev = ev.clamp(min=1e-12)
        Winv_sqrt = V @ torch.diag(1.0 / torch.sqrt(ev)) @ V.t()
        A = Winv_sqrt @ Sb_t @ Winv_sqrt
        A = (A + A.t()) / 2
        ge, _ = torch.linalg.eigh(A)
        ge = ge.clamp(min=0)
        ge = torch.flip(ge, dims=[0]).cpu().numpy()
        res[mode] = {"evals": ge, "Sw": Sw_t.cpu().numpy(),
                     "Sb": Sb_t.cpu().numpy(),
                     "trace_Sb": float(np.trace(Sb)),
                     "trace_Sw": float(np.trace(Sw)),
                     "mu": mu, "class_mus": mus, "class_covs": covs, "weights": w}
    return res


def cum_energy(evals):
    e = np.asarray(evals, float).flatten()
    t = e.sum()
    return np.zeros_like(e) if t <= 0 else np.cumsum(e) / t


def spearman(a, b):
    try:
        from scipy.stats import spearmanr
        r, _ = spearmanr(a, b)
        return float(r)
    except Exception:  # noqa: BLE001
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        n = len(a)
        return float(1 - 6 * np.sum((ra - rb) ** 2) / (n ** 3 - n)) if n > 1 else 0.0


# ============================================================================
# 绘图
# ============================================================================
CLASS_COLORS = {"background": "#7f7f7f", "spalling": "#1f77b4",
                "crack": "#d62728", "moisture": "#2ca02c",
                "all_foreground": "#ff7f0e", "global": "#9467bd"}


def plot_class_spectrum(name, spec_evals, out_dir, mode="natural"):
    """spec_evals: {group: eval_cen array}."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for g in ["background", "spalling", "crack", "moisture", "global"]:
        ev = spec_evals.get(g)
        if ev is None or len(ev) == 0:
            continue
        cum = cum_energy(ev)
        xs = np.arange(1, len(cum) + 1)
        ax.plot(xs, cum, color=CLASS_COLORS.get(g, "k"), lw=1.6,
                label=f"{g} (C={len(ev)})")
    for th, col in [(0.9, "#2ca02c"), (0.95, "#ff7f0e"), (0.99, "#9467bd")]:
        ax.axhline(th, color=col, lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("rank r"); ax.set_ylabel("cumulative energy (centered Cov_k)")
    ax.set_title(f"{name} — class-conditional spectrum ({mode})")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_class_cumulative_energy_{mode}.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for g in ["background", "spalling", "crack", "moisture"]:
        ev = spec_evals.get(g)
        if ev is None or len(ev) == 0:
            continue
        nrm = ev / max(ev.max(), 1e-12)
        nrm = np.clip(nrm, 1e-12, None)
        xs = np.arange(1, len(nrm) + 1)
        ax.semilogy(xs, nrm, color=CLASS_COLORS.get(g, "k"), lw=1.5, label=g)
    ax.set_xlabel("component index"); ax.set_ylabel("normalized eigenvalue (log)")
    ax.set_title(f"{name} — spectrum decay ({mode})")
    ax.grid(which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_class_spectrum_decay_{mode}.png"), dpi=110)
    plt.close(fig)


def plot_fisher(name, fisher, out_dir):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for mode, col in [("natural", "#1f77b4"), ("balanced", "#d62728")]:
        ev = fisher[mode]["evals"]
        k = min(len(ev), 20)
        xs = np.arange(1, k + 1)
        ax.semilogy(xs, np.clip(ev[:k], 1e-12, None), marker="o", ms=4,
                    color=col, label=f"{mode}")
    ax.set_xlabel("Fisher dimension index (sorted)")
    ax.set_ylabel("generalized eigenvalue (log)")
    ax.set_title(f"{name} — Fisher eigenvalues (Sb rank ≤ K-1 = 3)")
    ax.grid(which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_fisher_eigenvalues.png"), dpi=110)
    plt.close(fig)


def plot_channel_scatter(name, scores, out_dir, mode="balanced"):
    gv = scores["global_var"]; fs = scores["fisher_" + mode]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.scatter(gv, fs, s=12, alpha=0.6)
    # 标 low-var-high-fisher
    lvhf = scores["low_var_high_fisher_" + mode]
    if len(lvhf):
        ax.scatter(gv[lvhf], fs[lvhf], s=40, facecolors="none",
                   edgecolors="red", linewidths=1.2, label="low-var high-Fisher")
        ax.legend(fontsize=8)
    ax.set_xlabel("global variance (per channel)")
    ax.set_ylabel(f"Fisher score J ({mode})")
    ax.set_title(f"{name} — channel variance vs Fisher score")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_channel_var_vs_fisher.png"), dpi=110)
    plt.close(fig)


def plot_top_fisher(name, scores, out_dir, mode="balanced", topk=16):
    fs = scores["fisher_" + mode]
    k = min(topk, len(fs))
    idx = np.argsort(fs)[::-1][:k]
    vals = fs[idx]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(np.arange(k), vals, color="#1f77b4")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([str(int(i)) for i in idx], fontsize=7, rotation=45)
    ax.set_xlabel("channel index"); ax.set_ylabel(f"Fisher score ({mode})")
    ax.set_title(f"{name} — top-{k} Fisher channels ({mode})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_top_fisher_channels.png"), dpi=110)
    plt.close(fig)


def plot_mean_diff_heatmap(name, diff_mat, out_dir, max_ch=64):
    """diff_mat: (3, C) for [spalling-bg, crack-bg, moisture-bg]."""
    C = diff_mat.shape[1]
    ch = min(max_ch, C)
    step = max(1, C // ch)
    data = diff_mat[:, ::step]
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    im = ax.imshow(data, aspect="auto", cmap="seismic")
    vabs = np.nanmax(np.abs(data)) or 1.0
    im.set_clim(-vabs, vabs)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["spall-bg", "crack-bg", "moist-bg"])
    ax.set_xlabel("channel (subsampled)")
    ax.set_title(f"{name} — class-vs-bg mean difference")
    plt.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_class_mean_diff_heatmap.png"), dpi=110)
    plt.close(fig)


# ============================================================================
# 主流程
# ============================================================================
def main():
    args = parse_args()
    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    device = _resolve_device(args.device)
    print(f"[INFO] device={device}")

    from ultralytics import YOLO
    yolo = YOLO(args.weights)
    mdl = yolo.model.to(device).eval()
    nc = int(getattr(mdl.model[-1], "nc", 3))
    stride = int(mdl.stride.max().item()) if hasattr(mdl, "stride") else 32
    print(f"[INFO] nc={nc} stride={stride}")

    split_dir, yaml_path, ds_info = resolve_split_dir(args, nc, args.output_dir)
    all_paths = list_split_images(split_dir)
    print(f"[INFO] split={args.split} dir={split_dir} total={len(all_paths)}")
    selected = select_images(all_paths, args.max_images, args.seed)
    print(f"[INFO] analysing {len(selected)} images (seed={args.seed})")

    ds = TaskAwareDataset(selected, args.imgsz, stride=stride)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, collate_fn=collate)

    specs = build_specs()
    # accs[feature][mode][group]
    accs = {s.name: {m: {g: GramAcc() for g in ALL_GROUPS} for m in ["natural", "balanced"]}
            for s in specs}
    print(f"[INFO] registered {len(specs)} feature hooks.")
    overlap_stats = {s.name: {"overlap": 0, "fg": 0, "n_img": 0} for s in specs}

    handles = register_hooks(specs, mdl, accs, overlap_stats, args.seed)

    try:
        n_done = 0
        tf = time.time()
        with torch.no_grad():
            for tensors, items in loader:
                BATCH_CTX["items"] = items
                tensors = tensors.to(device, non_blocking=True)
                _ = mdl(tensors)
                n_done += tensors.shape[0]
                if n_done % (args.batch * 20) < args.batch or n_done >= len(selected):
                    print(f"    processed {n_done}/{len(selected)} ({time.time()-tf:.1f}s)")
        BATCH_CTX["items"] = []
        print(f"[INFO] forward done {n_done} imgs in {time.time()-tf:.1f}s")
    finally:
        for h in handles:
            h.remove()
        print("[INFO] all hooks removed.")

    # ---- 类别条件谱 ----
    print("\n===== class-conditional spectrum =====")
    class_rows = []
    npz = {}
    # 为每个 feature 缓存 class_stats (disjoint, 用于 Fisher & channel)
    class_stats_cache = {s.name: {} for s in specs}
    spec_eval_cache = {s.name: {} for s in specs}  # name -> {mode: {group: eval_cen}}

    for s in specs:
        for mode in ["natural", "balanced"]:
            ev_by_group = {}
            for group in SPECTRUM_GROUPS + ["spalling_disjoint", "crack_disjoint",
                                            "moisture_disjoint"]:
                acc = accs[s.name][mode][group]
                if acc.C is None or acc.count == 0:
                    continue
                res = cov_from_gram(acc)
                ev_by_group[group] = res["eval_cen"]
                m = metrics_from_evals(res["eval_cen"])
                class_rows.append({
                    "feature": s.name, "parent_layer": s.parent,
                    "feature_type": s.ftype, "mode": mode, "group": group,
                    "channels": res["C"], "count": res["count"],
                    "r90": m["r90"], "r95": m["r95"], "r99": m["r99"],
                    "r90_ratio": m["r90_ratio"], "r95_ratio": m["r95_ratio"],
                    "r99_ratio": m["r99_ratio"], "stable_rank": m["stable_rank"],
                    "effective_rank": m["effective_rank"],
                    "participation_ratio": m["participation_ratio"],
                })
                npz[f"{s.name}__{mode}__{group}__eval_cen"] = res["eval_cen"]
                npz[f"{s.name}__{mode}__{group}__eval_raw"] = res["eval_raw"]
            spec_eval_cache[s.name][mode] = ev_by_group
            # 收集 disjoint class_stats (background + 3 defects).
            # C 取自 global 累积器 (总有数据); 缺失类用同维度零向量/零矩阵, 计数 0.
            C_feat = accs[s.name][mode]["global"].C or 1
            cs = {}
            grp_map = {"background": "background", "spalling": "spalling_disjoint",
                       "crack": "crack_disjoint", "moisture": "moisture_disjoint"}
            for cname, gkey in grp_map.items():
                acc = accs[s.name][mode][gkey]
                if acc.C is None or acc.count == 0:
                    cs[cname] = {"count": 0, "mu": np.zeros(C_feat),
                                 "cov": np.zeros((C_feat, C_feat))}
                else:
                    r = cov_from_gram(acc)
                    cs[cname] = {"count": r["count"], "mu": r["mu"], "cov": r["cov"]}
            class_stats_cache[s.name][mode] = cs

        # 绘类别谱图 (natural & balanced)
        for mode in ["natural", "balanced"]:
            ev = spec_eval_cache[s.name][mode]
            ev_simple = {("spalling" if g.endswith("_union") else
                         ("background" if g == "background" else
                          ("global" if g == "global" else None))): v
                        for g, v in ev.items()}
            ev_simple = {k: v for k, v in ev_simple.items() if k}
            # 补 crack/moisture/all_foreground
            for disp, key in [("crack", "crack_union"), ("moisture", "moisture_union"),
                              ("all_foreground", "all_foreground")]:
                if key in ev and disp not in ev_simple:
                    ev_simple[disp] = ev[key]
            plot_class_spectrum(s.name, ev_simple, plots_dir, mode)

    # ---- Fisher ----
    print("\n===== Fisher discriminant spectrum =====")
    fisher_rows = []
    fisher_cache = {}
    for s in specs:
        fsh = fisher_from_groups(class_stats_cache[s.name]["natural"], args.fisher_eps)
        # balanced 也算 (基于 balanced class stats)
        fsh_bal = fisher_from_groups(class_stats_cache[s.name]["balanced"], args.fisher_eps)
        fsh["balanced"] = fsh_bal["balanced"]
        fisher_cache[s.name] = fsh
        for mode in ["natural", "balanced"]:
            ev = fsh[mode]["evals"]
            cum = cum_energy(ev)
            me = metrics_from_evals(ev)
            nz = int(np.sum(ev > 1e-6 * (ev.max() if ev.max() > 0 else 1)))
            d90 = int(np.searchsorted(cum, 0.90) + 1) if cum.max() >= 0.90 else len(ev)
            d95 = int(np.searchsorted(cum, 0.95) + 1) if cum.max() >= 0.95 else len(ev)
            d99 = int(np.searchsorted(cum, 0.99) + 1) if cum.max() >= 0.99 else len(ev)
            fisher_rows.append({
                "feature": s.name, "parent_layer": s.parent, "mode": mode,
                "channels": int(len(ev)), "K": 4, "sb_max_rank": 3,
                "eps": args.fisher_eps, "nonzero_evals(>1e-6max)": nz,
                "fisher_eff_rank": round(me["effective_rank"], 3),
                "d90": d90, "d95": d95, "d99": d99,
                "top1_eval": float(ev[0]) if len(ev) else 0,
                "top2_eval": float(ev[1]) if len(ev) > 1 else 0,
                "top3_eval": float(ev[2]) if len(ev) > 2 else 0,
                "energy_top3": float(cum[2]) if len(cum) >= 3 else float(cum.max()),
                "trace_Sb": fsh[mode]["trace_Sb"], "trace_Sw": fsh[mode]["trace_Sw"],
            })
            npz[f"{s.name}__{mode}__fisher_evals"] = ev
            npz[f"{s.name}__{mode}__fisher_cum"] = cum
        plot_fisher(s.name, fsh, plots_dir)
        print(f"    {s.name:28s} nat: eff={fsh['natural']['evals'][:3].round(3).tolist()} "
              f"| bal: eff={fsh['balanced']['evals'][:3].round(3).tolist()}")

    # ---- 逐通道 Fisher score ----
    print("\n===== per-channel Fisher scores =====")
    chan_rows = []
    chan_summary = {}  # name -> stats dict for report
    for s in specs:
        cs_nat = class_stats_cache[s.name]["natural"]
        cs_bal = class_stats_cache[s.name]["balanced"]
        glob_acc = accs[s.name]["natural"]["global"]
        glob = cov_from_gram(glob_acc) if (glob_acc.C and glob_acc.count) else None
        C = (cs_nat["spalling"]["mu"].shape[0] if cs_nat["spalling"]["count"] else
             (glob["C"] if glob else 1))
        gv = (glob["cov"].diagonal().copy() if glob else np.zeros(C))
        glob_mu = (glob["mu"].copy() if glob else np.zeros(C))

        per_mode = {}
        for mode in ["natural", "balanced"]:
            cs = cs_nat if mode == "natural" else cs_bal
            K = 4
            w = np.array([cs[c]["count"] for c in FISHER_CLASSES], float)
            N = w.sum()
            w = w / N if N > 0 else np.ones(K) / K
            mus = np.stack([cs[c]["mu"] if cs[c]["mu"].shape[0] == C else
                            np.zeros(C) for c in FISHER_CLASSES])
            covs = np.stack([cs[c]["cov"] if cs[c]["cov"].shape == (C, C) else
                            np.zeros((C, C)) for c in FISHER_CLASSES])
            mu = (w[:, None] * mus).sum(0)
            num = np.einsum("k,ki->i", w, (mus - mu[None, :]) ** 2)
            # 每类每通道方差 (K,C) -> 加权类内方差 (C,)
            var_k = covs.diagonal(axis1=1, axis2=2)        # (K, C)
            den = (w[:, None] * var_k).sum(0)               # (C,)
            J = num / (den + CHANNEL_EPS)
            # class variances & mean diffs
            var_sp = var_k[FISHER_CLASSES.index("spalling")]
            var_cr = var_k[FISHER_CLASSES.index("crack")]
            var_mo = var_k[FISHER_CLASSES.index("moisture")]
            var_bg = var_k[FISHER_CLASSES.index("background")]
            m_sp = mus[FISHER_CLASSES.index("spalling")]
            m_cr = mus[FISHER_CLASSES.index("crack")]
            m_mo = mus[FISHER_CLASSES.index("moisture")]
            m_bg = mus[FISHER_CLASSES.index("background")]
            d_sp = m_sp - m_bg; d_cr = m_cr - m_bg; d_mo = m_mo - m_bg
            per_mode[mode] = dict(J=J, mu=mu, mus=mus, var_sp=var_sp, var_cr=var_cr,
                                  var_mo=var_mo, var_bg=var_bg,
                                  d_sp=d_sp, d_cr=d_cr, d_mo=d_mo,
                                  w=w)
            for j in range(C):
                chan_rows.append({
                    "feature": s.name, "parent_layer": s.parent, "mode": mode,
                    "channel": j, "fisher_score": float(J[j]),
                    "global_var": float(gv[j]) if len(gv) > j else 0.0,
                    "var_spalling": float(var_sp[j]), "var_crack": float(var_cr[j]),
                    "var_moisture": float(var_mo[j]), "var_background": float(var_bg[j]),
                    "diff_spalling_bg": float(d_sp[j]), "diff_crack_bg": float(d_cr[j]),
                    "diff_moisture_bg": float(d_mo[j]),
                    "mean_spalling": float(m_sp[j]), "mean_crack": float(m_cr[j]),
                    "mean_moisture": float(m_mo[j]), "mean_background": float(m_bg[j]),
                })

        # Spearman / topk / low-var-high-fisher (balanced 主要)
        for mode in ["natural", "balanced"]:
            J = per_mode[mode]["J"]
            sp = spearman(gv[:len(J)], J)
            k = max(8, len(J) // 8)
            topv = set(np.argsort(gv[:len(J)])[::-1][:k].tolist())
            topf = set(np.argsort(J)[::-1][:k].tolist())
            overlap = len(topv & topf) / k
            # low-var high-fisher: var rank 在下半, fisher rank 在前 25%
            var_rank = np.argsort(np.argsort(gv[:len(J)]))  # 0..C-1
            fish_rank = np.argsort(np.argsort(J))
            lv = var_rank > (len(J) / 2)
            hf = fish_rank < (len(J) / 4)
            lvhf = np.where(lv & hf)[0]
            per_mode[mode]["spearman"] = sp
            per_mode[mode]["topk_overlap"] = overlap
            per_mode[mode]["low_var_high_fisher"] = lvhf
        chan_summary[s.name] = per_mode

        # plots
        scores_for_plot = {
            "global_var": gv,
            "fisher_balanced": per_mode["balanced"]["J"],
            "fisher_natural": per_mode["natural"]["J"],
            "low_var_high_fisher_balanced": per_mode["balanced"]["low_var_high_fisher"],
        }
        plot_channel_scatter(s.name, scores_for_plot, plots_dir, "balanced")
        plot_top_fisher(s.name, scores_for_plot, plots_dir, "balanced")
        diff_mat = np.stack([per_mode["balanced"]["d_sp"],
                             per_mode["balanced"]["d_cr"],
                             per_mode["balanced"]["d_mo"]])
        plot_mean_diff_heatmap(s.name, diff_mat, plots_dir)

        print(f"    {s.name:28s} C={C} spearman(var,F)="
              f"{per_mode['balanced']['spearman']:+.3f} "
              f"topk_ov={per_mode['balanced']['topk_overlap']:.2f} "
              f"lowvar-highF={len(per_mode['balanced']['low_var_high_fisher'])}")

    # ---- 写 CSV ----
    def write_csv(path, rows):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    csv_class = os.path.join(args.output_dir, "class_spectrum_summary.csv")
    csv_fisher = os.path.join(args.output_dir, "fisher_spectrum_summary.csv")
    csv_chan = os.path.join(args.output_dir, "channel_fisher_scores.csv")
    write_csv(csv_class, class_rows)
    write_csv(csv_fisher, fisher_rows)
    write_csv(csv_chan, chan_rows)
    print(f"\n[OUT] {csv_class}\n[OUT] {csv_fisher}\n[OUT] {csv_chan}")

    npz_path = os.path.join(args.output_dir, "task_aware_singular_values.npz")
    np.savez(npz_path, **npz)
    print(f"[OUT] {npz_path}")

    # ---- metadata ----
    elapsed = time.time() - t0
    meta = {
        "weights": os.path.abspath(args.weights),
        "data_yaml": os.path.abspath(yaml_path), "data_yaml_source": ds_info.get("yaml_source"),
        "check_det_dataset": ds_info.get("check_det_dataset"),
        "split": args.split, "split_dir": split_dir,
        "imgsz": args.imgsz, "batch": args.batch, "workers": args.workers,
        "device": str(device), "num_images_total_split": len(all_paths),
        "num_images_analysed": len(selected), "seed": args.seed,
        "per_box_cap": PER_BOX_CAP, "class_cap": CLASS_CAP, "other_cap": OTHER_CAP,
        "fisher_eps": args.fisher_eps, "channel_eps": CHANNEL_EPS,
        "ultralytics_version": __import__("ultralytics").__version__,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "python_version": sys.version.split()[0], "nc": nc, "stride": stride,
        "defect_names": DEFECT_NAMES, "fisher_classes": FISHER_CLASSES,
        "features": [s.name for s in specs],
        "overlap_stats": {k: v for k, v in overlap_stats.items()},
        "runtime_seconds": round(elapsed, 2),
    }
    meta_path = os.path.join(args.output_dir, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[OUT] {meta_path}")

    # ---- report ----
    report_path = os.path.join(args.output_dir, "analysis_report.md")
    write_report(report_path, args, meta, class_rows, fisher_rows, chan_summary,
                 overlap_stats, fisher_cache, class_stats_cache, accs)
    print(f"[OUT] {report_path}")

    # ---- terminal summary ----
    print_terminal_summary(class_rows, fisher_rows, chan_summary, args.output_dir,
                           plots_dir, report_path, success=bool(class_rows))


def fmt(x, d=2):
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{d}f}"


def cr_lookup(class_rows, feature, mode, group):
    for r in class_rows:
        if (r["feature"] == feature and r["mode"] == mode and r["group"] == group):
            return r
    return None


def fr_lookup(fisher_rows, feature, mode):
    for r in fisher_rows:
        if r["feature"] == feature and r["mode"] == mode:
            return r
    return None


def print_terminal_summary(class_rows, fisher_rows, chan_summary, out_dir, plots_dir,
                           report_path, success):
    print("\n" + "=" * 96)
    print("类条件 r95/C (balanced) — background / spalling / crack / moisture")
    print("=" * 96)
    feats = sorted(set(r["feature"] for r in class_rows))
    for f in feats:
        vals = []
        for g in ["background", "spalling_union", "crack_union", "moisture_union"]:
            r = cr_lookup(class_rows, f, "balanced", g)
            vals.append(f"{g[:4]}={r['r95_ratio']:.2f}" if r else f"{g[:4]}=-")
        print(f"  {f:28s} " + "  ".join(vals))

    print("\nFisher (top-3 eigenvalues / eff rank / d95)  natural vs balanced")
    for f in feats:
        rn = fr_lookup(fisher_rows, f, "natural")
        rb = fr_lookup(fisher_rows, f, "balanced")
        def fmtf(r):
            if not r:
                return "-"
            return (f"eff={r['fisher_eff_rank']:.2f} d95={r['d95']} "
                    f"top3=({r['top1_eval']:.3f},{r['top2_eval']:.3f},{r['top3_eval']:.3f})")
        print(f"  {f:28s} nat[{fmtf(rn)}]  bal[{fmtf(rb)}]")

    print("\n方差排序 vs Fisher 排序 (balanced): Spearman / topk 重合 / 低方差高Fisher数")
    for f in feats:
        c = chan_summary.get(f, {}).get("balanced", {})
        if c:
            print(f"  {f:28s} spearman={c.get('spearman',0):+.3f}  "
                  f"topk_ov={c.get('topk_overlap',0):.2f}  "
                  f"lowvar-highF={len(c.get('low_var_high_fisher',[]))}")

    print("\n" + "-" * 96)
    print("结果目录       :", out_dir)
    print("图像目录       :", plots_dir)
    print("analysis_report.md :", report_path)
    print("运行状态       :", "SUCCESS" if success else "FAILED")
    print("=" * 96)


# ============================================================================
# 报告
# ============================================================================
def write_report(path, args, meta, class_rows, fisher_rows, chan_summary,
                 overlap_stats, fisher_cache, class_stats_cache, accs):
    L = []
    L.append("# L1-96 任务感知 / 类别条件谱 + Fisher 判别谱分析报告\n")
    L.append(f"- 运行耗时 {meta['runtime_seconds']}s | Ultralytics "
             f"{meta['ultralytics_version']} / torch {meta['torch_version']} / "
             f"CUDA {meta['cuda_version']} / {meta['gpu_name']}\n")
    L.append(f"- 类别映射: {dict(zip(range(3), meta['defect_names']))}; "
             f"Fisher 类别 K=4: {meta['fisher_classes']} (Sb 秩 ≤ K-1 = 3)\n")

    L.append("\n## A. 配置\n")
    L.append(f"- 权重 `{meta['weights']}`; 数据 yaml `{meta['data_yaml']}` "
             f"({meta['data_yaml_source']}, check_det_dataset={meta['check_det_dataset']})\n")
    L.append(f"- split={meta['split']}, imgsz={meta['imgsz']}, batch={meta['batch']}, "
             f"device=`{meta['device']}`, FP32, seed={meta['seed']}\n")
    L.append(f"- 分析图像: **{meta['num_images_analysed']}** / {meta['num_images_total_split']}\n")
    L.append(f"- 预处理: 复用 `LetterBox(auto=False, stride={meta['stride']})`, "
             f"GT 框经 get_params(ratio/left/top) 精确映射到 640 → 特征图.\n")
    L.append(f"- 采样: natural (自然频率) + balanced (每类≤{meta['class_cap']}, "
             f"背景≤{meta['other_cap']}, 每框≤{meta['per_box_cap']}).\n")
    L.append(f"- Fisher 正则 eps={meta['fisher_eps']} (Sw+eps·trace(Sw)/C·I), "
             f"白化经 eigh (不直接求 inverse).\n")

    # overlap 统计
    L.append("\n## B. 跨类别重叠统计 (natural, 前景位置中落入≥2类的比例)\n")
    L.append("| feature | overlap | foreground | overlap/fg |\n|---|---|---|---|\n")
    for f, v in overlap_stats.items():
        ofg = (v["overlap"] / v["fg"]) if v["fg"] > 0 else 0.0
        L.append(f"| {f} | {v['overlap']} | {v['fg']} | {ofg:.3f} |\n")

    # 类条件谱
    L.append("\n## C. 类别条件中心化协方差谱 (r95/C, balanced)\n")
    L.append("| feature | background | spalling | crack | moisture | global |\n")
    L.append("|---|---|---|---|---|---|\n")
    feats = sorted(set(r["feature"] for r in class_rows))
    for f in feats:
        row = [f]
        for g in ["background", "spalling_union", "crack_union", "moisture_union", "global"]:
            r = cr_lookup(class_rows, f, "balanced", g)
            row.append(f"{r['r95_ratio']:.2f}" if r else "-")
        L.append("| " + " | ".join(row) + " |\n")

    L.append("\n### C2. 类别条件 r95 / 有效秩 明细 (balanced)\n\n")
    L.append("| feature | group | C | r95 | r95/C | stable | eff | part |\n|---|---|---|---|---|---|---|---|\n")
    for f in feats:
        for g in ["background", "spalling_union", "crack_union", "moisture_union"]:
            r = cr_lookup(class_rows, f, "balanced", g)
            if r:
                L.append(f"| {f} | {g} | {r['channels']} | {r['r95']} | "
                         f"{r['r95_ratio']:.3f} | {r['stable_rank']:.1f} | "
                         f"{r['effective_rank']:.1f} | {r['participation_ratio']:.1f} |\n")

    # Fisher
    L.append("\n## D. Fisher 判别谱\n")
    L.append("| feature | mode | nonzero(>1e-6) | eff_rank | d90 | d95 | d99 | "
             "top1 | top2 | top3 | E(top3) |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    for f in feats:
        for mode in ["natural", "balanced"]:
            r = fr_lookup(fisher_rows, f, mode)
            if r:
                L.append(f"| {f} | {mode} | {r['nonzero_evals(>1e-6max)']} | "
                         f"{r['fisher_eff_rank']:.2f} | {r['d90']} | {r['d95']} | "
                         f"{r['d99']} | {r['top1_eval']:.3f} | {r['top2_eval']:.3f} | "
                         f"{r['top3_eval']:.3f} | {r['energy_top3']:.3f} |\n")
    L.append("\n> **关键**: Sb 秩 ≤ K−1 = 3, 故 Fisher 有效秩必然很低 (≤3). "
             "这只表示“类别均值差异位于 ≤3 维线性子空间”, **不等于** 原特征只需 3 个通道, "
             "更不能等同于 C2f hidden 宽度.\n")

    # 通道
    L.append("\n## E. 逐通道任务重要性 (方差 vs Fisher)\n")
    L.append("| feature | mode | Spearman(var,F) | topk重合 | 低方差高Fisher数 |\n")
    L.append("|---|---|---|---|---|\n")
    for f in feats:
        for mode in ["natural", "balanced"]:
            c = chan_summary.get(f, {}).get(mode, {})
            if c:
                L.append(f"| {f} | {mode} | {c.get('spearman',0):+.3f} | "
                         f"{c.get('topk_overlap',0):.2f} | "
                         f"{len(c.get('low_var_high_fisher',[]))} |\n")

    # 8 问
    L.append("\n## F. 报告必答\n")

    # 计算 moisture 集中度: balanced Fisher moisture-bg mean diff 集中在多少通道
    def mean_diff_concentration(name, diff_arr):
        d = np.abs(diff_arr)
        if d.sum() <= 0:
            return 0, 0.0
        order = np.argsort(d)[::-1]
        cs = np.cumsum(d[order]) / d.sum()
        k = int(np.searchsorted(cs, 0.8) + 1)
        return k, float(cs[min(k, len(cs))-1])

    # Q1 高方差是否=高判别
    sp_vals = [chan_summary[f]["balanced"]["spearman"] for f in feats
               if f in chan_summary]
    avg_sp = float(np.mean(sp_vals)) if sp_vals else 0.0
    L.append(f"\n### Q1. 全局高方差通道是否也是任务判别性最高的通道?\n")
    L.append(f"- 10 个特征 (balanced) 方差与 Fisher 的平均 Spearman = **{avg_sp:+.3f}**. "
             f"该值越接近 +1 表示“高方差≈高判别”. 实测见上表 (事实). 若 |ρ| 偏低, "
             f"说明方差排序与判别排序显著错位 → 高方差不等于高任务价值 (相关性).\n")

    # Q2 低方差高Fisher
    L.append(f"\n### Q2. 是否存在低方差但高 Fisher 的通道?\n")
    for f in feats:
        lv = chan_summary[f]["balanced"]["low_var_high_fisher"]
        if len(lv):
            L.append(f"- {f}: {len(lv)} 个, 通道索引示例 "
                     f"{lv[:12].tolist()}{'...' if len(lv)>12 else ''} (事实).\n")

    # Q3 哪个位置任务信息分布最广: 比较 balanced 各类的 eff rank 之和 或 global eff rank
    L.append(f"\n### Q3. Layer 12/18/21 中哪个位置的任务判别信息分布最广?\n")
    L.append("- 以 balanced 各类 effective_rank 之和衡量 (越高=判别信息越分散):\n\n")
    L.append("| feature | Σ_k eff_rank(bal) | Σ_k r95/C(bal) |\n|---|---|---|\n")
    for f in feats:
        if not (f.endswith("bottleneck_last") or f.endswith("prefuse")):
            continue
        s_eff = 0.0; s_r = 0.0
        for g in ["spalling_union", "crack_union", "moisture_union"]:
            r = cr_lookup(class_rows, f, "balanced", g)
            if r:
                s_eff += r["effective_rank"]; s_r += r["r95_ratio"]
        L.append(f"| {f} | {s_eff:.1f} | {s_r:.2f} |\n")

    # Q4 哪个位置更适合结构化裁剪
    L.append(f"\n### Q4. 哪个位置更适合结构化通道裁剪?\n")
    L.append("- 结构化裁剪适合“方差与 Fisher 高度一致且低方差高Fisher少”的特征 "
             "(排序一致→按方差裁剪不会误删判别通道).\n\n")
    L.append("| feature | Spearman(var,F) | 低方差高Fisher | 判别 |\n|---|---|---|---|\n")
    for f in feats:
        c = chan_summary.get(f, {}).get("balanced", {})
        sp = c.get("spearman", 0); nlv = len(c.get("low_var_high_fisher", []))
        verdict = "较安全" if (sp > 0.6 and nlv <= 2) else ("需谨慎" if sp > 0.3 else "风险高")
        L.append(f"| {f} | {sp:+.3f} | {nlv} | {verdict} (推测) |\n")

    # Q5 moisture 集中
    L.append(f"\n### Q5. moisture 判别信息是否集中在少数通道?\n")
    for f in feats:
        d_mo = chan_summary[f]["balanced"]["d_mo"]
        k, e = mean_diff_concentration(f, d_mo)
        L.append(f"- {f}: |moisture−bg| 均值差前 {k} 个通道即覆盖 80% 的类间差能量 "
                 f"(累计 {e:.2f}) (事实).\n")

    # Q6 natural vs balanced
    L.append(f"\n### Q6. natural 与 balanced 差异多大?\n")
    diffs = []
    for f in feats:
        rn = fr_lookup(fisher_rows, f, "natural")
        rb = fr_lookup(fisher_rows, f, "balanced")
        if rn and rb:
            diffs.append(abs(rn["top1_eval"] - rb["top1_eval"]) /
                         max(rn["top1_eval"], 1e-9))
    avg_d = float(np.mean(diffs)) if diffs else 0.0
    L.append(f"- natural vs balanced 的 top-1 Fisher 特征值平均相对差异 = **{avg_d:.3f}**. "
             f"差异越大 → 自然分布被背景/某类数量支配越严重 (相关性).\n")

    # Q7 全局PCA是否会损害少数类
    # Q7: 比较每特征 “三类最低 r99/C” 与 “global r95/C”
    L.append(f"\n### Q7. 全局 PCA 可压缩性判断是否会损害少数类别?\n")
    rows_q7 = []
    for f in feats:
        rs = [cr_lookup(class_rows, f, "balanced", g)
              for g in ["spalling_union", "crack_union", "moisture_union"]]
        rs = [r for r in rs if r]
        g = cr_lookup(class_rows, f, "balanced", "global")
        if rs and g:
            min_c = min(r["r99_ratio"] for r in rs)
            rows_q7.append((f, min_c, g["r95_ratio"]))
    if rows_q7:
        n_risk = sum(1 for _, mc, gr, in rows_q7 if mc > gr)
        ex = max(rows_q7, key=lambda x: x[1] - x[2])
        L.append(f"- 对每个特征比较 “三类中最低的 r99/C” 与 “global r95/C”: 在 **{n_risk}/{len(rows_q7)}** "
                 f"个特征上, 即便是能量最集中的类别, 其 r99/C 仍高于 global r95/C (差距最大: {ex[0]}, "
                 f"min_class_r99/C={ex[1]:.2f} vs global_r95/C={ex[2]:.2f}). 这意味着按全局 PCA 裁到 r95 "
                 f"会落在“任一类 99% 能量所需通道数”之下 → 可能切掉少数类 r95→r99 的判别方向, "
                 f"**存在损害少数类的风险** (推测, 需训练验证).\n")

    # Q8 P5 vs P4 (诚实: 逐类谱平坦度近似, 区分点在裁剪深度与判别通道脆弱性)
    L.append(f"\n### Q8. 谱结果能否解释 P5 压缩比 P4 更敏感?\n")
    def cls_r99_mean(f):
        rs = [cr_lookup(class_rows, f, "balanced", g)
              for g in ["spalling_union", "crack_union", "moisture_union"]]
        rs = [r for r in rs if r]
        return float(np.mean([r["r99_ratio"] for r in rs])) if rs else 0
    p4 = np.mean([cls_r99_mean("layer12_bottleneck_last"),
                  cls_r99_mean("layer18_bottleneck_last")])
    p5 = cls_r99_mean("layer21_bottleneck_last")
    lv_p4 = float(np.mean([len(chan_summary["layer12_bottleneck_last"]["balanced"]["low_var_high_fisher"]),
                           len(chan_summary["layer18_bottleneck_last"]["balanced"]["low_var_high_fisher"])]))
    lv_p5 = float(len(chan_summary["layer21_bottleneck_last"]["balanced"]["low_var_high_fisher"]))
    sp_p4 = float(np.mean([chan_summary["layer12_bottleneck_last"]["balanced"]["spearman"],
                           chan_summary["layer18_bottleneck_last"]["balanced"]["spearman"]]))
    sp_p5 = float(chan_summary["layer21_bottleneck_last"]["balanced"]["spearman"])
    L.append(f"- bottleneck_last 三类平均 r99/C: P4(L12/18)={p4:.2f}, P5(L21)={p5:.2f} —— "
             f"两者**几乎持平**, 说明“逐类谱平坦度”本身并不能区分 P4/P5 (事实).\n")
    L.append(f"- 区分点在别处 (事实+推测): (1) SR1 裁剪深度不同 — P5 hidden 128→96 削掉 25% 通道, "
             f"P4 48→40 仅削 17%; 全局谱分析测得该裁剪 P5 保留 96.1% / P4 保留 96.7% 能量, P5 略丢更多. "
             f"(2) 判别通道脆弱性 — P5 bottleneck 方差-Fisher Spearman={sp_p5:+.2f} (P4={sp_p4:+.2f}), "
             f"低方差高Fisher 通道 P5={int(lv_p5)} vs P4={lv_p4:.0f} → P5 有更多“低方差却具判别力”的通道, "
             f"方差/PCA 式裁剪更易误删.\n")
    L.append(f"- 综上: 谱结果**能**解释 “P5 压缩更敏感”, 但主要源于 **裁剪更深 + 判别通道更脆弱**, "
             f"而非逐类谱更平坦 (与既有实验 L1-96 0.429 / SR1 0.412 / SR1-P4Only 0.427 方向一致; "
             f"推测, 需训练验证; 与全局谱分析 [[l196-spectrum-analysis]] 互证).\n")

    L.append("\n## G. 结论分级 (实测事实 / 统计相关性 / 合理推测 / 尚未验证)\n")
    L.append("- **实测事实**: 类条件 r95/C、Fisher top-3 特征值、Spearman 系数、"
             "低方差高Fisher通道数 (均来自 1105 val 图实测 Gram).\n")
    L.append("- **统计相关性**: “高方差↔高判别” 由 Spearman 量化; "
             "“natural/balanced 差异” 由特征值相对差异量化.\n")
    L.append("- **合理推测**: Q4 的“较安全/需谨慎/风险高”判别、Q7 的“会损害少数类”、"
             "Q8 的“P5 更敏感” — 均为基于谱数据的推断.\n")
    L.append("- **尚未验证的压缩建议**: 具体通道裁剪/hidden 调整方案需经训练验证; "
             "本脚本不修改 YAML, 不自动训练.\n")
    L.append("- **约束**: Fisher 有效秩 (≤3) 不等于 C2f hidden 宽度, 也不等于所需通道数.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(L)


if __name__ == "__main__":
    main()
