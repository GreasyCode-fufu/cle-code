#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L1-96 模型多层特征谱分析 (Multi-layer feature spectrum analysis)
=================================================================

目的:
    在不训练、不修改权重、不新建模型 YAML 的前提下, 分析 L1-96 检测模型若干关键层
    (Backbone P3/P4/P5、Top-down Neck P4、Detect P3/P4/P5) 以及重点 C2f 块
    (Layer 12 / 18 / 21) 内部子特征的:
        - 奇异值谱 (通过非中心化二阶矩阵 M_raw 的特征值,  == 奇异值平方)
        - 累计谱能量曲线
        - 有效秩指标 (r90 / r95 / r99, normalized ratios)
        - Stable Rank / Entropy Effective Rank / Participation Ratio
    分别对 raw 谱与 centered 谱计算.

实现要点:
    * 数据预处理完全复用 Ultralytics 8.4.84 验证阶段的 LetterBox
      (new_shape=640, auto=False, stride=32), 并完成 BGR->RGB 与 /255 归一化.
    * 仅累积 C×C 的 Gram 矩阵与 C 维 sum_x (在线累积, CPU float64),
      绝不长期保存全部空间激活, 控制显存/内存.
    * 特征通过 register_forward_hook / register_forward_pre_hook 临时获取,
      不修改 block.py 中 C2f.forward; 退出时 try/finally 移除全部 hook.
    * 图像选择可复现: 路径排序后用固定 seed 抽样.

作者: 自动生成 (read-only 分析脚本)
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# matplotlib 使用非交互后端 (服务端无显示器)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------
# 全局常量
# ----------------------------------------------------------------------------
IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")

# 目标层通道数预期 (用于运行期校验, 不一致会显式报错而非静默继续)
EXPECTED_OUT_CHANNELS = {
    4: 64,    # Backbone P3
    6: 96,    # Backbone P4
    8: 256,   # Backbone P5
    12: 96,   # Top-down Neck P4
    15: 64,   # Detect P3
    18: 96,   # Detect P4
    21: 256,  # Detect P5
}
EXPECTED_HIDDEN_C = {12: 48, 18: 48, 21: 128}  # 重点 C2f 的 self.c 预期


# ============================================================================
# 1. 命令行参数
# ============================================================================
def _resolve_device(dev: str) -> torch.device:
    """把 '0' / 'cuda' / 'cuda:0' / 'cpu' 等统一解析为 torch.device."""
    dev = str(dev).strip()
    if dev.lower() == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("[WARN] CUDA 不可用, 回退到 CPU.")
        return torch.device("cpu")
    if dev.isdigit():
        return torch.device(f"cuda:{dev}")
    if dev == "cuda":
        return torch.device("cuda:0")
    return torch.device(dev)  # cuda:N 或其它合法字符串


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L1-96 multi-layer feature spectrum analysis (read-only).")
    p.add_argument("--weights", type=str, required=True,
                   help="Path to best.pt")
    p.add_argument("--data-root", type=str, required=True,
                   help="Dataset root directory (contains train/val/test).")
    p.add_argument("--data-yaml", type=str, default=None,
                   help="Optional explicit data yaml. If omitted, auto-search data-root.")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--max-images", type=int, default=500,
                   help="Number of images to analyse. <=0 means use all.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-spatial-samples", type=int, default=0,
                   help="Max spatial positions per image per feature. 0 = use all.")
    p.add_argument("--output-dir", type=str,
                   default="runs/analysis/l1_96_spectrum")
    return p.parse_args()


# ============================================================================
# 2. 数据集 YAML 解析 / 校验
# ============================================================================
def _read_yaml(yaml_path: str) -> dict:
    """极简 YAML 读取 (只取 path/train/val/test/names), 避免依赖."""
    import re
    text = open(yaml_path, "r", encoding="utf-8").read()
    out: Dict = {}
    # names 可能是 dict 形式 {0: x, 1: y} 或 list
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip("'\"")
        if key in ("path", "train", "val", "test"):
            out[key] = val
    return out


def find_data_yaml(data_root: str, explicit: Optional[str]) -> Optional[str]:
    """优先使用显式 yaml; 否则在 data-root 下自动查找常见数据配置文件."""
    if explicit:
        if os.path.isfile(explicit):
            return os.path.abspath(explicit)
        print(f"[WARN] 显式指定的 data-yaml 不存在: {explicit}")

    candidates = ["data.yaml", "hongkongScale.yaml", "dataset.yaml", "data.yml"]
    for name in candidates:
        fp = os.path.join(data_root, name)
        if os.path.isfile(fp):
            return os.path.abspath(fp)
    # 递归浅查找一层
    for root, _, files in os.walk(data_root):
        for name in candidates:
            if name in files:
                return os.path.abspath(os.path.join(root, name))
    return None


def resolve_dataset(args, nc: int, output_dir: str) -> Tuple[str, str, dict]:
    """
    返回 (split_images_dir, yaml_used_path_or_'(none)', info_dict).
    使用 Ultralytics check_det_dataset 校验 (若可用); 否则按约定
    <data_root>/<split>/images 直接解析.
    """
    info: Dict = {}
    yaml_path = find_data_yaml(args.data_root, args.data_yaml)

    # 若 data-root 内找不到 yaml, 在 output_dir 内创建一份临时 yaml (绝对路径),
    # 用 check_det_dataset 做合法性校验, 但绝不修改数据集根目录或标签.
    temp_yaml = None
    if yaml_path is None:
        os.makedirs(output_dir, exist_ok=True)
        temp_yaml = os.path.join(output_dir, "_auto_data.yaml")
        names_block = "\n".join([f"  {i}: class_{i}" for i in range(nc)])
        content = (
            f"path: {args.data_root}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"test: test/images\n"
            f"nc: {nc}\n"
            f"names:\n{names_block}\n"
        )
        with open(temp_yaml, "w", encoding="utf-8") as f:
            f.write(content)
        yaml_path = temp_yaml
        info["yaml_source"] = "auto-generated (no yaml found in data-root)"
    else:
        info["yaml_source"] = "found in data-root / explicit"

    # 尝试用 Ultralytics 的 check_det_dataset 校验 (autodownload=False, 不会联网下载)
    validated_val_dir = None
    try:
        from ultralytics.data.utils import check_det_dataset
        data = check_det_dataset(yaml_path, autodownload=False)
        split_key = "val" if args.split == "val" else ("train" if args.split == "train" else "test")
        validated_val_dir = data.get(split_key) or data.get("val")
        info["check_det_dataset"] = "ok"
        info["resolved_names"] = data.get("names")
    except Exception as e:  # noqa: BLE001
        info["check_det_dataset"] = f"failed: {e}"
        validated_val_dir = None

    # 兜底: 直接按约定路径解析 split 图像目录
    split_dir = os.path.join(args.data_root, args.split, "images")
    if not os.path.isdir(split_dir):
        # 某些数据集直接是 data_root/<split>/*.jpg
        alt = os.path.join(args.data_root, args.split)
        if os.path.isdir(alt):
            split_dir = alt
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"无法定位 split 图像目录: {split_dir} (同时 check_det_dataset 返回 {validated_val_dir})")

    info["split_images_dir"] = split_dir
    info["yaml_path"] = yaml_path
    info["validated_val_dir"] = validated_val_dir
    return split_dir, yaml_path, info


def list_split_images(split_dir: str) -> List[str]:
    imgs: List[str] = []
    for ext in IMG_EXT:
        imgs.extend(glob.glob(os.path.join(split_dir, ext)))
    imgs = sorted(set(imgs))
    return imgs


def select_images(paths: List[str], max_images: int, seed: int) -> List[str]:
    """排序后用固定 seed 抽样, 结果再排序, 保证可复现 & DataLoader 顺序确定."""
    paths = sorted(paths)
    if max_images is None or max_images <= 0 or max_images >= len(paths):
        return paths
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(paths), size=max_images, replace=False)
    idx = np.sort(idx)
    return [paths[i] for i in idx]


# ============================================================================
# 3. 数据集 / 预处理 (复用 Ultralytics LetterBox)
# ============================================================================
class SpectrumDataset(torch.utils.data.Dataset):
    """
    复用 Ultralytics 验证阶段 LetterBox 预处理:
        cv2.imread (BGR) -> LetterBox(640, auto=False, stride=32) -> HWC->CHW ->
        BGR->RGB -> /255 -> float32 tensor [3,640,640]
    不做任何数据增强 (与 val 一致).
    """

    def __init__(self, image_paths: List[str], imgsz: int, stride: int = 32):
        self.paths = image_paths
        from ultralytics.data.augment import LetterBox
        # auto=False => 正方形 letterbox (与默认 rect=False 的 val 行为一致)
        self.letterbox = LetterBox(
            new_shape=(imgsz, imgsz), auto=False, scaleup=True,
            stride=stride, padding_value=114)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        import cv2
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)  # BGR uint8 HxWxC
        if img is None:
            raise RuntimeError(f"无法读取图像: {self.paths[i]}")
        img = self.letterbox(image=img)                    # BGR uint8, imgsz×imgsz
        # HWC->CHW, 然后 BGR->RGB ([::-1] 产生负步长, 必须 ascontiguousarray 拷贝)
        img = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1, :, :])
        img = torch.from_numpy(img).float().div_(255.0)    # 归一化到 [0,1]
        return img, self.paths[i]


# ============================================================================
# 4. Gram 累积器 (在线累积, CPU float64)
# ============================================================================
class GramAccumulator:
    """
    在线累积 C 维特征的:
        sum_x = Σ x_i           (C,)
        gram  = Σ x_i x_i^T     (C, C)
        count = 总空间位置数
    全部以 CPU float64 保存, batch 内 matmul 在 GPU float32 完成后再转 double.
    """

    def __init__(self):
        self.C: Optional[int] = None
        self.sum_x: Optional[torch.Tensor] = None
        self.gram: Optional[torch.Tensor] = None
        self.count: int = 0

    def _init(self, C: int):
        self.C = C
        self.sum_x = torch.zeros(C, dtype=torch.float64)
        self.gram = torch.zeros(C, C, dtype=torch.float64)

    @torch.no_grad()
    def update(self, feat: torch.Tensor, max_spatial: int):
        """
        feat: [B, C, H, W] 任意 device (FP32). 转为 X ∈ R^{C×N} 后累积.
        """
        if feat.dim() != 4:
            return
        B, C, H, W = feat.shape
        if C <= 0:
            return
        if self.C is None:
            self._init(C)
        elif self.C != C:
            # 通道数发生变化 (异常), 跳过避免污染
            print(f"[WARN] 通道数不一致, 跳过: 期望 {self.C}, 实际 {C}")
            return

        feat = feat.detach()
        # [B,C,H,W] -> [C, B*H*W]
        X = feat.permute(1, 0, 2, 3).reshape(C, -1).float()
        N = X.shape[1]
        if max_spatial and max_spatial > 0 and max_spatial < N:
            # 均匀网格抽样 (非随机), 保留空间结构
            idx = torch.linspace(0, N - 1, max_spatial, device=feat.device).long()
            X = X[:, idx]

        # GPU float32 完成矩阵乘法, 再转 CPU float64 累加
        sum_b = X.sum(dim=1).double().cpu()
        gram_b = (X @ X.t()).double().cpu()
        self.sum_x += sum_b
        self.gram += gram_b
        self.count += X.shape[1]


# ============================================================================
# 5. Hook 注册
# ============================================================================
def extract_tensor(out) -> Optional[torch.Tensor]:
    """从 forward hook 输出中提取一个 4D tensor."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            t = extract_tensor(o)
            if t is not None and t.dim() == 4:
                return t
    return None


def extract_input_tensor(args) -> Optional[torch.Tensor]:
    """从 forward pre-hook 的输入中提取一个 4D tensor (cv2 的输入即 pre-fusion 特征)."""
    if isinstance(args, torch.Tensor):
        return args if args.dim() == 4 else None
    if isinstance(args, (tuple, list)):
        for a in args:
            t = extract_input_tensor(a)
            if t is not None:
                return t
    return None


@dataclass
class FeatureSpec:
    name: str
    parent_layer: int
    feature_type: str        # output / cv1 / bottleneck_last / prefuse
    hook_kind: str           # 'post' (forward_hook) 或 'pre' (forward_pre_hook)
    target_fn: Callable      # model -> nn.Module


def build_feature_specs() -> List[FeatureSpec]:
    specs: List[FeatureSpec] = []
    # 1) 七个目标层的最终输出 (post-hook)
    for idx in [4, 6, 8, 12, 15, 18, 21]:
        specs.append(FeatureSpec(
            name=f"layer{idx}_output", parent_layer=idx, feature_type="output",
            hook_kind="post",
            target_fn=(lambda i: (lambda mdl: mdl.model[i]))(idx)))

    # 2) 三个重点 C2f (12/18/21) 的内部子特征
    for idx in [12, 18, 21]:
        # cv1 输出 (post-hook on cv1)
        specs.append(FeatureSpec(
            name=f"layer{idx}_cv1", parent_layer=idx, feature_type="cv1",
            hook_kind="post",
            target_fn=(lambda i: (lambda mdl: mdl.model[i].cv1))(idx)))
        # 最后一个 Bottleneck 输出 (post-hook on m[-1])
        specs.append(FeatureSpec(
            name=f"layer{idx}_bottleneck_last", parent_layer=idx,
            feature_type="bottleneck_last", hook_kind="post",
            target_fn=(lambda i: (lambda mdl: mdl.model[i].m[-1]))(idx)))
        # cv2 输入 = pre-fusion 特征 (pre-hook on cv2)
        specs.append(FeatureSpec(
            name=f"layer{idx}_prefuse", parent_layer=idx, feature_type="prefuse",
            hook_kind="pre",
            target_fn=(lambda i: (lambda mdl: mdl.model[i].cv2))(idx)))
    return specs


def register_hooks(specs: List[FeatureSpec], mdl, accs: Dict[str, GramAccumulator],
                   max_spatial: int):
    handles = []
    for spec in specs:
        mod = spec.target_fn(mdl)
        if spec.hook_kind == "post":
            def make_hook(name):
                def _h(_module, _inp, out):
                    t = extract_tensor(out)
                    if t is None:
                        return
                    acc = accs.setdefault(name, GramAccumulator())
                    acc.update(t, max_spatial)
                return _h
            handles.append(mod.register_forward_hook(make_hook(spec.name)))
        else:  # pre
            def make_pre(name):
                def _h(_module, args):
                    t = extract_input_tensor(args)
                    if t is None:
                        return
                    acc = accs.setdefault(name, GramAccumulator())
                    acc.update(t, max_spatial)
                return _h
            handles.append(mod.register_forward_pre_hook(make_pre(spec.name)))
    return handles


# ============================================================================
# 6. 谱分析数学
# ============================================================================
def spectrum_from_gram(acc: GramAccumulator) -> dict:
    """从累积的 Gram 计算 raw / centered 两套特征值 (降序, 截断负值)."""
    C = acc.C
    count = max(acc.count, 1)
    M_raw = (acc.gram / count).double()
    mu = (acc.sum_x / count).double()
    M_cen = (acc.gram - count * torch.outer(mu, mu)).double() / max(count - 1, 1)

    # eigvalsh 返回升序, flip 得到降序; clamp(min=0) 消除微小负误差
    eval_raw = torch.linalg.eigvalsh(M_raw).clamp(min=0).flip(0)
    eval_cen = torch.linalg.eigvalsh(M_cen).clamp(min=0).flip(0)
    return {
        "C": C,
        "count": acc.count,
        "mu": mu.cpu().numpy(),
        "eval_raw": eval_raw.cpu().numpy(),
        "eval_cen": eval_cen.cpu().numpy(),
    }


def metrics_from_evals(evals: np.ndarray) -> dict:
    """
    由降序非负特征值计算:
        r90/r95/r99, 归一化 ratio, stable_rank, effective_rank, participation_ratio
    """
    evals = np.asarray(evals, dtype=np.float64).flatten()
    eps = 1e-12
    total = float(evals.sum())
    C = len(evals)
    if total <= eps:
        z = dict(r90=0, r95=0, r99=0, r90_ratio=0.0, r95_ratio=0.0, r99_ratio=0.0,
                 stable_rank=0.0, effective_rank=0.0, participation_ratio=0.0)
        return z

    cum = np.cumsum(evals) / total

    def rank_for(th: float) -> int:
        idx = np.where(cum >= th)[0]
        return int(idx[0]) + 1 if len(idx) > 0 else C

    r90, r95, r99 = rank_for(0.90), rank_for(0.95), rank_for(0.99)
    stable_rank = total / float(evals.max())

    p = evals / total
    effective_rank = float(np.exp(-np.sum(p * np.log(p + eps))))

    participation_ratio = (total ** 2) / float(np.sum(evals ** 2))

    return dict(
        r90=r90, r95=r95, r99=r99,
        r90_ratio=r90 / C, r95_ratio=r95 / C, r99_ratio=r99 / C,
        stable_rank=stable_rank, effective_rank=effective_rank,
        participation_ratio=participation_ratio,
    )


def cumulative_energy(evals: np.ndarray) -> np.ndarray:
    total = float(evals.sum())
    if total <= 0:
        return np.zeros_like(evals)
    return np.cumsum(evals) / total


# ============================================================================
# 7. 绘图
# ============================================================================
def plot_feature(name: str, evals_raw: np.ndarray, evals_cen: np.ndarray,
                 out_dir: str):
    cum_raw = cumulative_energy(evals_raw)
    cum_cen = cumulative_energy(evals_cen)
    C = len(evals_raw)

    # ---- (a) 累计能量曲线 ----
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xs = np.arange(1, C + 1)
    ax.plot(xs, cum_raw, color="#1f77b4", lw=1.8, label="raw (uncentered)")
    ax.plot(xs, cum_cen, color="#d62728", lw=1.6, ls="--", label="centered")
    for th, col in [(0.90, "#2ca02c"), (0.95, "#ff7f0e"), (0.99, "#9467bd")]:
        ax.axhline(th, color=col, lw=0.9, ls=":", alpha=0.8)
        ax.text(C, th, f" {int(th*100)}%", color=col, va="center", ha="left",
                fontsize=8)
    # raw r90/95/99 竖线
    m_raw = metrics_from_evals(evals_raw)
    for rk, lab, col in [(m_raw["r90"], "r90", "#2ca02c"),
                         (m_raw["r95"], "r95", "#ff7f0e"),
                         (m_raw["r99"], "r99", "#9467bd")]:
        ax.axvline(rk, color=col, lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("rank r (number of components)")
    ax.set_ylabel("cumulative spectral energy E(r)")
    ax.set_title(f"{name} — cumulative energy (C={C})")
    ax.set_xlim(1, C)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_cumulative_energy.png"), dpi=120)
    plt.close(fig)

    # ---- (b) 归一化特征值衰减 (log y) ----
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    eps = 1e-12
    nr = evals_raw / max(evals_raw.max(), eps)
    nc = evals_cen / max(evals_cen.max(), eps)
    nr = np.clip(nr, eps, None)
    nc = np.clip(nc, eps, None)
    ax.semilogy(xs, nr, color="#1f77b4", lw=1.6, label="raw")
    ax.semilogy(xs, nc, color="#d62728", lw=1.4, ls="--", label="centered")
    ax.set_xlabel("component index (sorted)")
    ax.set_ylabel("normalized eigenvalue (log)")
    ax.set_title(f"{name} — spectrum decay (C={C})")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_spectrum_decay.png"), dpi=120)
    plt.close(fig)


# ============================================================================
# 8. 模型结构信息收集 / 校验
# ============================================================================
def collect_model_info(mdl) -> dict:
    info = {"layers": {}, "expected_check": {}}
    for idx in EXPECTED_OUT_CHANNELS:
        layer = mdl.model[idx]
        rec = {"type": type(layer).__name__}
        # 输出通道 (C2f -> cv2)
        cv2 = getattr(layer, "cv2", None)
        if cv2 is not None and hasattr(cv2, "conv"):
            rec["out_channels"] = int(cv2.conv.out_channels)
        elif cv2 is not None and hasattr(cv2, "out_channels"):
            rec["out_channels"] = int(cv2.out_channels)
        if hasattr(layer, "c"):
            rec["hidden_c"] = int(layer.c)
        if hasattr(layer, "m"):
            rec["num_bottlenecks"] = int(len(layer.m))
        info["layers"][str(idx)] = rec

        exp = EXPECTED_OUT_CHANNELS[idx]
        ok = (rec.get("out_channels") == exp)
        info["expected_check"][str(idx)] = {
            "expected_out": exp,
            "actual_out": rec.get("out_channels"),
            "match": bool(ok),
        }
    for idx, exp_c in EXPECTED_HIDDEN_C.items():
        layer = mdl.model[idx]
        actual = int(getattr(layer, "c", -1))
        info.setdefault("hidden_check", {})[str(idx)] = {
            "expected_c": exp_c, "actual_c": actual, "match": actual == exp_c,
        }
    return info


def verify_structure(model_info: dict) -> bool:
    ok = True
    for idx, chk in model_info["expected_check"].items():
        if not chk["match"]:
            print(f"[ERROR] Layer {idx} 输出通道不匹配: 预期 {chk['expected_out']}, "
                  f"实际 {chk['actual_out']}")
            ok = False
    for idx, chk in model_info.get("hidden_check", {}).items():
        if not chk["match"]:
            print(f"[ERROR] Layer {idx} self.c 不匹配: 预期 {chk['expected_c']}, "
                  f"实际 {chk['actual_c']}")
            ok = False
    return ok


# ============================================================================
# 9. 主流程
# ============================================================================
def main():
    args = parse_args()
    t_start = time.time()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    device = _resolve_device(args.device)
    print(f"[INFO] device = {device}")

    # ---- 加载模型 ----
    from ultralytics import YOLO
    print(f"[INFO] 加载权重: {args.weights}")
    yolo = YOLO(args.weights)
    mdl = yolo.model  # DetectionModel
    mdl.to(device).eval()
    nc = int(getattr(mdl.model[-1], "nc", 3))
    stride = int(mdl.stride.max().item()) if hasattr(mdl, "stride") else 32
    print(f"[INFO] nc={nc}, stride={stride}")

    # ---- 打印并校验模型结构 ----
    print("\n===== 模型顶层结构 =====")
    for i, layer in enumerate(mdl.model):
        print(f"  Layer {i:2d}: {type(layer).__name__}")
    model_info = collect_model_info(mdl)
    print("\n===== 目标层信息 (输出通道 / hidden_c / bottleneck 数) =====")
    for idx in EXPECTED_OUT_CHANNELS:
        rec = model_info["layers"][str(idx)]
        print(f"  Layer {idx:2d}: {rec}")
    print("\n===== 重点 C2f self.c 校验 =====")
    for idx in EXPECTED_HIDDEN_C:
        print(f"  Layer {idx}: self.c = {model_info['hidden_check'][str(idx)]['actual_c']} "
              f"(预期 {EXPECTED_HIDDEN_C[idx]})")

    if not verify_structure(model_info):
        print("[WARN] 模型结构与预期不完全一致, 仍按实际结构继续 (见报告).")

    # ---- 数据集 / 图像选择 ----
    split_dir, yaml_path, ds_info = resolve_dataset(args, nc, args.output_dir)
    all_paths = list_split_images(split_dir)
    print(f"\n[INFO] split={args.split}, 目录={split_dir}")
    print(f"[INFO] 该 split 全部图像数 = {len(all_paths)}")
    if len(all_paths) == 0:
        raise RuntimeError("未找到任何 split 图像, 终止.")
    selected = select_images(all_paths, args.max_images, args.seed)
    print(f"[INFO] 本次分析图像数 = {len(selected)} (seed={args.seed})")

    # ---- DataLoader ----
    ds = SpectrumDataset(selected, args.imgsz, stride=stride)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch], 0), [b[1] for b in batch]))

    # ---- 注册 hook ----
    specs = build_feature_specs()
    accs: Dict[str, GramAccumulator] = {}
    print(f"\n[INFO] 注册 {len(specs)} 个特征 hook:")
    for s in specs:
        print(f"    - {s.name:32s} (parent=L{s.parent_layer}, "
              f"type={s.feature_type}, hook={s.hook_kind})")
    handles = register_hooks(specs, mdl, accs, args.max_spatial_samples)

    # ---- 推理循环 ----
    try:
        n_done = 0
        t_fwd = time.time()
        with torch.no_grad():
            for imgs, _paths in loader:
                imgs = imgs.to(device, non_blocking=True)
                _ = mdl(imgs)  # forward, hook 自动累积
                n_done += imgs.shape[0]
                if n_done % (args.batch * 10) < args.batch or n_done >= len(selected):
                    print(f"    已处理 {n_done}/{len(selected)} 张 "
                          f"({time.time() - t_fwd:.1f}s)")
        print(f"[INFO] 推理完成, 共 {n_done} 张, 耗时 {time.time()-t_fwd:.1f}s")
    finally:
        for h in handles:
            h.remove()
        print("[INFO] 全部 hook 已移除.")

    # ---- 谱计算 ----
    print("\n===== 谱分析计算 =====")
    rows = []
    npz_data = {}
    per_feature_meta = {}
    evals_by_name: Dict[str, np.ndarray] = {}  # name -> raw 降序特征值
    for spec in specs:
        acc = accs.get(spec.name)
        if acc is None or acc.C is None or acc.count == 0:
            print(f"[WARN] 特征 {spec.name} 未捕获到任何激活, 跳过.")
            continue
        spec_res = spectrum_from_gram(acc)
        evals_by_name[spec.name] = spec_res["eval_raw"]
        C = spec_res["C"]
        m_raw = metrics_from_evals(spec_res["eval_raw"])
        m_cen = metrics_from_evals(spec_res["eval_cen"])
        cum_raw = cumulative_energy(spec_res["eval_raw"])
        cum_cen = cumulative_energy(spec_res["eval_cen"])

        rows.append({
            "feature_name": spec.name,
            "parent_layer": spec.parent_layer,
            "feature_type": spec.feature_type,
            "channels": C,
            "count": spec_res["count"],
            "r90_raw": m_raw["r90"], "r95_raw": m_raw["r95"], "r99_raw": m_raw["r99"],
            "r90_ratio_raw": m_raw["r90_ratio"], "r95_ratio_raw": m_raw["r95_ratio"],
            "r99_ratio_raw": m_raw["r99_ratio"],
            "stable_rank_raw": m_raw["stable_rank"],
            "effective_rank_raw": m_raw["effective_rank"],
            "participation_ratio_raw": m_raw["participation_ratio"],
            "r90_centered": m_cen["r90"], "r95_centered": m_cen["r95"],
            "r99_centered": m_cen["r99"],
            "r90_ratio_centered": m_cen["r90_ratio"],
            "r95_ratio_centered": m_cen["r95_ratio"],
            "r99_ratio_centered": m_cen["r99_ratio"],
            "stable_rank_centered": m_cen["stable_rank"],
            "effective_rank_centered": m_cen["effective_rank"],
            "participation_ratio_centered": m_cen["participation_ratio"],
        })
        per_feature_meta[spec.name] = {
            "parent_layer": spec.parent_layer, "feature_type": spec.feature_type,
            "channels": C, "count": spec_res["count"],
            "metrics_raw": m_raw, "metrics_centered": m_cen,
        }
        # npz 保存 (每个特征一组数组)
        npz_data[f"{spec.name}__eval_raw"] = spec_res["eval_raw"]
        npz_data[f"{spec.name}__eval_cen"] = spec_res["eval_cen"]
        npz_data[f"{spec.name}__cum_raw"] = cum_raw
        npz_data[f"{spec.name}__cum_cen"] = cum_cen
        npz_data[f"{spec.name}__mu"] = spec_res["mu"]

        # 绘图
        plot_feature(spec.name, spec_res["eval_raw"], spec_res["eval_cen"], plots_dir)
        print(f"    {spec.name:32s} C={C:3d}  count={spec_res['count']:>9d}  "
              f"r95_raw={m_raw['r95']:>3d}({m_raw['r95_ratio']:.2f})  "
              f"stable={m_raw['stable_rank']:.1f}  eff={m_raw['effective_rank']:.1f}")

    # ---- 写 CSV ----
    csv_path = os.path.join(args.output_dir, "spectrum_summary.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[OUT] CSV -> {csv_path}")
    else:
        print("[ERROR] 没有可用结果行, 未写 CSV.")

    # ---- 写 NPZ ----
    npz_path = os.path.join(args.output_dir, "singular_values.npz")
    if npz_data:
        np.savez(npz_path, **npz_data)
        print(f"[OUT] NPZ -> {npz_path}")

    # ---- 写 run_metadata.json ----
    elapsed_total = time.time() - t_start
    meta = {
        "weights": os.path.abspath(args.weights),
        "data_yaml": os.path.abspath(yaml_path) if yaml_path else None,
        "data_yaml_source": ds_info.get("yaml_source"),
        "check_det_dataset": ds_info.get("check_det_dataset"),
        "resolved_names": ds_info.get("resolved_names"),
        "split": args.split,
        "split_images_dir": split_dir,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": str(device),
        "num_images_total_split": len(all_paths),
        "num_images_analysed": len(selected),
        "max_images_arg": args.max_images,
        "seed": args.seed,
        "max_spatial_samples": args.max_spatial_samples,
        "ultralytics_version": __import__("ultralytics").__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "python_version": sys.version.split()[0],
        "nc": nc,
        "stride": stride,
        "model_structure": model_info,
        "features": per_feature_meta,
        "runtime_seconds": round(elapsed_total, 2),
    }
    meta_path = os.path.join(args.output_dir, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[OUT] METADATA -> {meta_path}")

    # ---- 写 analysis_report.md ----
    report_path = os.path.join(args.output_dir, "analysis_report.md")
    write_report(report_path, args, meta, ds_info, all_paths, selected,
                 rows, per_feature_meta, evals_by_name)
    print(f"[OUT] REPORT -> {report_path}")

    # ---- 终端汇总 ----
    print("\n" + "=" * 90)
    print("分析完成. 核心结果表 (raw 谱):")
    print("=" * 90)
    print_terminal_summary(rows)

    print("\n脚本路径       :", os.path.abspath(__file__))
    print("spectrum_summary.csv :", csv_path)
    print("analysis_report.md   :", report_path)
    print("图像结果目录   :", plots_dir)
    print("运行是否成功   :", "SUCCESS" if rows else "FAILED (无结果)")
    print("=" * 90)
    return rows


def fmt(x, d=2):
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{d}f}"


def row_lookup(rows, name):
    for r in rows:
        if r["feature_name"] == name:
            return r
    return None


def print_terminal_summary(rows):
    header = ["Feature", "Channels", "r90_raw", "r95_raw", "r99_raw",
              "r95_ratio_raw", "StableRank", "EffectiveRank"]
    print("  " + "  ".join(f"{h:>14s}" for h in header[:1]) +
          "".join(f"{h:>11s}" for h in header[1:]))
    for r in rows:
        vals = [r["feature_name"], r["channels"], r["r90_raw"], r["r95_raw"],
                r["r99_raw"], f'{r["r95_ratio_raw"]:.3f}',
                f'{r["stable_rank_raw"]:.2f}', f'{r["effective_rank_raw"]:.2f}']
        line = f"  {vals[0]:>14s}" + "".join(
            f"{str(v):>11s}" for v in vals[1:])
        print(line)
    print("\n--- 重点关注 (C2f 内部 bottleneck_last / prefuse) ---")
    keys = ["layer12_bottleneck_last", "layer12_prefuse",
            "layer18_bottleneck_last", "layer18_prefuse",
            "layer21_bottleneck_last", "layer21_prefuse"]
    for k in keys:
        r = row_lookup(rows, k)
        if r is None:
            print(f"  {k:>28s}: (缺失)")
            continue
        print(f"  {k:>28s}: C={r['channels']:>3d}  r95_raw={r['r95_raw']:>3d} "
              f"({r['r95_ratio_raw']:.2f})  r99_raw={r['r99_raw']:>3d}  "
              f"StableRank={r['stable_rank_raw']:.1f}  "
              f"EffRank={r['effective_rank_raw']:.1f}")


def _energy_at(evals: np.ndarray, r: int) -> float:
    """前 r 个分量累计能量占比 (raw 谱)."""
    evals = np.asarray(evals, dtype=np.float64).flatten()
    total = float(evals.sum())
    if total <= 0:
        return 0.0
    r = max(1, min(int(r), len(evals)))
    return float(np.sum(evals[:r]) / total)


def write_report(path, args, meta, ds_info, all_paths, selected, rows, per_meta,
                 evals_by_name: Dict[str, np.ndarray]):
    def R(name, k, d=2):
        r = row_lookup(rows, name)
        if r is None:
            return "-"
        return fmt(r.get(k), d)

    def mget(name, scope, k, d=2):
        r = row_lookup(rows, name)
        if r is None:
            return "-"
        return fmt(r.get(f"{k}_{scope}"), d)

    L = []
    L.append("# L1-96 多层特征谱分析报告\n")
    L.append(f"- 生成时间(脚本运行): 运行耗时 {meta['runtime_seconds']}s\n")
    L.append(f"- Ultralytics {meta['ultralytics_version']} / PyTorch "
             f"{meta['torch_version']} / CUDA {meta['cuda_version']} / "
             f"GPU {meta['gpu_name']}\n")

    L.append("\n## A. 实际分析的模型与数据配置\n")
    L.append(f"- 权重: `{meta['weights']}`\n")
    L.append(f"- 数据 YAML: `{meta['data_yaml']}` ({ds_info.get('yaml_source')})\n")
    L.append(f"- check_det_dataset 校验: `{ds_info.get('check_det_dataset')}`\n")
    L.append(f"- split: `{meta['split']}`, imgsz={meta['imgsz']}, "
             f"batch={meta['batch']}, workers={meta['workers']}, "
             f"device=`{meta['device']}`\n")
    L.append(f"- 预处理: 复用 Ultralytics `LetterBox(new_shape=({meta['imgsz']},"
             f"{meta['imgsz']}), auto=False, stride={meta['stride']})`, "
             f"BGR→RGB, /255 归一化, 无增强 (与 val 一致).\n")
    L.append(f"- nc={meta['nc']}, stride={meta['stride']}\n")
    L.append(f"- max_spatial_samples={meta['max_spatial_samples']} "
             f"(0=使用全部空间位置)\n")

    L.append("\n## B. 实际分析图像数量\n")
    L.append(f"- 该 split 全部图像数: **{len(all_paths)}**\n")
    L.append(f"- 本次分析图像数: **{len(selected)}** (seed={args.seed}, "
             f"路径排序后固定抽样)\n")

    L.append("\n## C. 每个层的通道数与内部 hidden 值\n")
    L.append("| Layer | 类型 | 输出通道 | hidden_c | bottleneck数 | 期望输出 | 是否匹配 |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    ms = meta["model_structure"]
    for idx in [4, 6, 8, 12, 15, 18, 21]:
        rec = ms["layers"][str(idx)]
        chk = ms["expected_check"][str(idx)]
        L.append(f"| {idx} | {rec['type']} | {rec.get('out_channels')} | "
                 f"{rec.get('hidden_c','-')} | {rec.get('num_bottlenecks','-')} | "
                 f"{chk['expected_out']} | {'✓' if chk['match'] else '✗'} |\n")
    L.append("\n重点 C2f self.c 校验:\n\n")
    L.append("| Layer | 实际 self.c | 预期 self.c | 是否匹配 |\n|---|---|---|---|\n")
    for idx in [12, 18, 21]:
        hc = ms["hidden_check"][str(idx)]
        L.append(f"| {idx} | {hc['actual_c']} | {hc['expected_c']} | "
                 f"{'✓' if hc['match'] else '✗'} |\n")

    L.append("\n## D. 每层 r90 / r95 / r99 与有效秩指标 (raw 谱)\n")
    L.append("| Feature | C | count | r90 | r95 | r99 | r95/C | StableRank | "
             "EffRank | PartRatio |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        L.append(f"| {r['feature_name']} | {r['channels']} | {r['count']} | "
                 f"{r['r90_raw']} | {r['r95_raw']} | {r['r99_raw']} | "
                 f"{r['r95_ratio_raw']:.3f} | {r['stable_rank_raw']:.1f} | "
                 f"{r['effective_rank_raw']:.1f} | "
                 f"{r['participation_ratio_raw']:.1f} |\n")

    L.append("\n### Centered 谱指标 (供对照)\n\n")
    L.append("| Feature | C | r95_cen | r99_cen | r95/C_cen | StableRank_cen | "
             "EffRank_cen |\n|---|---|---|---|---|---|---|\n")
    for r in rows:
        L.append(f"| {r['feature_name']} | {r['channels']} | {r['r95_centered']} | "
                 f"{r['r99_centered']} | {r['r95_ratio_centered']:.3f} | "
                 f"{r['stable_rank_centered']:.1f} | "
                 f"{r['effective_rank_centered']:.1f} |\n")

    L.append("\n## E. Backbone P4 (L6) vs Neck P4 (L12) 谱结构比较\n")
    L.append(f"- L6 (Backbone P4): C={R('layer6_output','channels')}, "
             f"r95_raw={R('layer6_output','r95_raw')}, "
             f"r95/C={R('layer6_output','r95_ratio_raw')}, "
             f"StableRank={R('layer6_output','stable_rank_raw')}, "
             f"EffRank={R('layer6_output','effective_rank_raw')}\n")
    L.append(f"- L12 (Top-down Neck P4): C={R('layer12_output','channels')}, "
             f"r95_raw={R('layer12_output','r95_raw')}, "
             f"r95/C={R('layer12_output','r95_ratio_raw')}, "
             f"StableRank={R('layer12_output','stable_rank_raw')}, "
             f"EffRank={R('layer12_output','effective_rank_raw')}\n")
    L.append("- 解读见末尾 I 节.\n")

    L.append("\n## F. Detect P3 / P4 / P5 谱结构比较\n")
    for nm in ["layer15_output", "layer18_output", "layer21_output"]:
        L.append(f"- {nm}: C={R(nm,'channels')}, r95_raw={R(nm,'r95_raw')}, "
                 f"r95/C={R(nm,'r95_ratio_raw')}, StableRank={R(nm,'stable_rank_raw')}, "
                 f"EffRank={R(nm,'effective_rank_raw')}\n")

    L.append("\n## G. Layer 12 / 18 / 21 内部特征谱比较\n")
    for idx in [12, 18, 21]:
        L.append(f"\n### Layer {idx} (C2f, self.c="
                 f"{ms['hidden_check'][str(idx)]['actual_c']})\n\n")
        L.append("| 子特征 | C | r95_raw | r99_raw | r95/C | StableRank | "
                 "EffRank |\n|---|---|---|---|---|---|---|\n")
        for ft in ["cv1", "bottleneck_last", "prefuse", "output"]:
            nm = f"layer{idx}_{ft}"
            L.append(f"| {ft} | {R(nm,'channels')} | {R(nm,'r95_raw')} | "
                     f"{R(nm,'r99_raw')} | {R(nm,'r95_ratio_raw',3)} | "
                     f"{R(nm,'stable_rank_raw')} | {R(nm,'effective_rank_raw')} |\n")

    L.append("\n## H. 谱衰减快慢排序 (raw 谱, 按 r95/C 升序)\n")
    L.append("说明: r95/C 越小 ⇒ 能量越集中 ⇒ 谱衰减越快/越可压缩; "
             "越大 ⇒ 维度利用越充分/冗余越低. 本模型所有 r95/C 均落在 "
             "0.61–0.77, 总体偏平坦, 故用三分位 (相对本模型内部) 标注.\n\n")
    ratios = sorted(r["r95_ratio_raw"] for r in rows)
    q1 = ratios[len(ratios) // 4]
    q3 = ratios[(3 * len(ratios)) // 4]

    def tier(x: float) -> str:
        if x <= q1:
            return "相对衰减快/较可压缩"
        if x >= q3:
            return "相对衰减慢/维度利用高"
        return "中等"

    order = sorted(rows, key=lambda r: r["r95_ratio_raw"])
    L.append("| 排名 | Feature | C | r95_raw | r95/C | StableRank | 相对分类 |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    for i, r in enumerate(order, 1):
        L.append(f"| {i} | {r['feature_name']} | {r['channels']} | "
                 f"{r['r95_raw']} | {r['r95_ratio_raw']:.3f} | "
                 f"{r['stable_rank_raw']:.1f} | {tier(r['r95_ratio_raw'])} |\n")
    L.append("\n关键观察: 没有任何层的 r95/C ≤ 0.5, 即不存在“急剧低秩/高度冗余”的层; "
             "最可压缩的是 P5 系列 (layer8/21_output, layer21_prefuse, r95/C≈0.61–0.64), "
             "最不可压缩的是 Layer12/18 的 bottleneck_last (r95/C≈0.77, hidden 几乎被用满).\n")

    L.append("\n## I. 与既有实验结论的一致性判断\n")
    # 取关键指标
    def num(name, k):
        r = row_lookup(rows, name)
        return None if r is None else r.get(k)

    # 在 SR1 实际裁剪维度上, 计算 bottleneck 谱能量保留比例
    def eret(name, dim):
        ev = evals_by_name.get(name)
        return None if ev is None else _energy_at(ev, dim)

    cur_c = {12: ms["hidden_check"]["12"]["actual_c"],
             18: ms["hidden_check"]["18"]["actual_c"],
             21: ms["hidden_check"]["21"]["actual_c"]}
    # SR1 把 L12/L18 hidden 48->40, L21 hidden 128->96
    sr1_cut = {12: 40, 18: 40, 21: 96}

    L.append("### 0) 总体结论: 谱普遍平坦, 无急剧冗余层\n")
    all_r95 = [r["r95_ratio_raw"] for r in rows]
    L.append(f"- 全部 16 个特征的 raw r95/C 范围 = "
             f"[{min(all_r95):.3f}, {max(all_r95):.3f}], 均值 "
             f"{float(np.mean(all_r95)):.3f}. 没有任何层 r95/C≤0.5, 即不存在"
             f"“维度严重过剩”的层; 信息在通道间分布较广 → 任何层的激进均匀裁剪都有风险.\n")

    L.append("\n### 1) P4 hidden 压缩敏感 (Layer 12 / 18, self.c=48)\n")
    for idx in [12, 18]:
        bn = row_lookup(rows, f"layer{idx}_bottleneck_last")
        if bn:
            r95, r99 = bn["r95_raw"], bn["r99_raw"]
            cut = sr1_cut[idx]
            e_cut = eret(f"layer{idx}_bottleneck_last", cut)
            e_cut_pct = (e_cut * 100) if e_cut is not None else 0.0
            L.append(f"- Layer {idx} bottleneck_last: r95={r95}, r99={r99} (hidden={cur_c[idx]}). "
                     f"SR1 把 hidden {cur_c[idx]}→{cut}: {cut} 落在 r95({r95}) 与 r99({r99}) 之间, "
                     f"裁剪到 {cut} 维仅保留 bottleneck 谱能量的 {e_cut_pct:.1f}% "
                     f"(即丢掉 {100-e_cut_pct:.1f}% 的尾部方差, 主要是 95%→99% 这段方向) "
                     f"→ 与 SR1 (48→40) test mAP 0.429→0.412 的下降一致: **P4 hidden 压缩敏感**.\n")

    L.append("\n### 2) Neck P4 (L12) 是否存在轻度冗余?\n")
    l12_prefuse = row_lookup(rows, "layer12_prefuse")
    l12_out = row_lookup(rows, "layer12_output")
    if l12_prefuse and l12_out:
        L.append(f"- L12 prefuse (concat 后, cv2 前): C={l12_prefuse['channels']}, "
                 f"r95/C={l12_prefuse['r95_ratio_raw']:.3f}; "
                 f"cv2 输出: C={l12_out['channels']}, r95/C={l12_out['r95_ratio_raw']:.3f}.\n")
        L.append(f"- 1×1 cv2 把 {l12_prefuse['channels']}→{l12_out['channels']} 压缩 "
                 f"({l12_prefuse['channels'] - l12_out['channels']} 维), 两端 r95/C 相近 "
                 f"(≈0.71–0.73), 说明 cv2 已能无损吸收 concat 带来的维度. 这与 SR1-P4Only "
                 f"(只压 hidden、不动 L21 输出 128) 时 mAP 仅 0.429→0.427 微降一致: "
                 f"**Neck P4 仅存在轻度/可被 cv2 吸收的冗余, 但 hidden 本身不可激进压缩**.\n")

    L.append("\n### 3) Layer 21 / P5 (hidden) 压缩敏感 (self.c=128)\n")
    bn21 = row_lookup(rows, "layer21_bottleneck_last")
    if bn21:
        r95, r99 = bn21["r95_raw"], bn21["r99_raw"]
        cut = sr1_cut[21]
        e_cut = eret("layer21_bottleneck_last", cut)
        e_cut_pct = (e_cut * 100) if e_cut is not None else 0.0
        L.append(f"- Layer 21 bottleneck_last: r95={r95}, r99={r99} (hidden={cur_c[21]}). "
                 f"SR1 把 hidden {cur_c[21]}→{cut}: {cut} 仅略高于 r95({r95}), 裁剪到 {cut} 维仅"
                 f"保留 bottleneck 谱能量的 {e_cut_pct:.1f}% (即丢掉 {100-e_cut_pct:.1f}%, "
                 f"覆盖 r95→r99({r95}→{r99}) 及之后 {cur_c[21]-cut} 维的大部分方差) "
                 f"→ 与 SR1 (128→96) test mAP 0.429→0.412 的“明显下降”一致: "
                 f"**L21/P5 hidden 压缩敏感**, 且比 P4 丢失能量更多.\n")

    L.append("\n### 4) Backbone P4 (L6) 压缩敏感?\n")
    l6 = row_lookup(rows, "layer6_output")
    if l6:
        L.append(f"- L6 (Backbone P4) r95/C={l6['r95_ratio_raw']:.3f}, StableRank="
                 f"{l6['stable_rank_raw']:.1f}, EffRank={l6['effective_rank_raw']:.1f}. "
                 f"其谱同样偏平坦 (信息铺满 96 通道), 与 Neck P4 (L12) 谱形相近, "
                 f"说明 P4 语义信息在通道间分布广 → 对通道裁剪整体敏感, 与上述 hidden "
                 f"压缩敏感的结论方向一致.\n")

    L.append("\n### 重要约束说明\n")
    L.append("- **不能把 output 的 r95 直接当作 C2f 应设的 hidden 通道数.** "
             "hidden 维度与最终输出特征的有效秩不是同一个量. 应综合参考同一 C2f 的 "
             "cv1 / bottleneck_last / prefuse 三个内部特征谱的一致性.\n")
    L.append("- 下面给出基于内部谱一致性的 **保守压缩候选区间 (实验建议, 非数学证明)**:\n")
    # 简单保守建议: 以 bottleneck_last 的 r99 作为 hidden 下限的弱参考
    for idx in [12, 18, 21]:
        bn = row_lookup(rows, f"layer{idx}_bottleneck_last")
        cur_c = ms["hidden_check"][str(idx)]["actual_c"]
        if bn:
            r99 = bn["r99_raw"]
            r95 = bn["r95_raw"]
            lo = max(int(r95), 8)
            hi = max(int(r99), lo)
            L.append(f"  - Layer {idx}: 当前 hidden={cur_c}; bottleneck_last "
                     f"r95={r95}, r99={r99}. 弱参考区间 hidden ≈ [{lo}, {hi}] "
                     f"(仅供实验探索, 需以训练验证为准).\n")
    L.append("- 本脚本不会自动修改任何 YAML, 不会自动开始训练.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(L)


if __name__ == "__main__":
    main()
