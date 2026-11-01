"""
CLE fair-comparison TRAINING script (classifyML) — aligned to the CUFIT eval protocol.
Based on train.py, with ONLY these changes:
  1. Model chosen via --model (registry in cle_fair_common), not a hardcoded import.
  2. Best checkpoint / early stop selected by val MACRO-F1 (not val_acc).
  3. Saves run_meta.json (dataset mean/std, model, img_size...) so the fair eval
     uses identical normalization.
  4. Training progress bar uses tqdm.rich (golden bar, width = terminal x 0.618) to
     match evaluate_2class.py. train_one_epoch is a LOCAL copy here; its training logic
     (UNWEIGHTED CrossEntropy / AMP / AdamW) is identical to utils.train_one_epoch —
     only the progress-bar style differs. utils.py is NOT modified.
Unchanged from train.py: dataset self-computed mean/std (cached), train/val transforms,
AdamW + CosineAnnealingLR + AMP, patient-level train/val read from directories, --seed.
Usage:
  python train_cle_fair.py --model fast_t0 --num_classes 2 --img_size 224 --seed 42 \
      --data-path-train /path/cle/train --data-path-val /path/cle/val \
      --save-dir ./weights --epochs 300
"""
import os
import re
import sys
import csv
import json
import time
import shutil
import random
import argparse
import warnings
import subprocess
from pathlib import Path

import torch
import numpy as np
import torch.optim as optim
from datetime import datetime, timedelta
from torchvision import transforms
from tqdm.rich import tqdm as rich_tqdm
from rich.console import Console

from my_dataset import MyDataSet
from utils import read_split_train_data, read_split_val_data
# Reuse train.py's dataset mean/std machinery verbatim (no modification to train.py):
from train import get_or_compute_mean_std, _norm_transform_desc
from cle_fair_common import build_model, build_val_transform, MODEL_REGISTRY
from consensus_manifest import filter_training_samples, sha256_file

from sklearn.metrics import f1_score, accuracy_score


CLASSIFIER_ROOTS = {"head", "fc", "classifier"}

# 各模型"新增模块"允许缺失的参数段名：ImageNet 预训练权重中不存在这些结构，
# 随机初始化属于设计预期，不视为加载失败。段名与参数路径节点精确匹配。
MODEL_ALLOWED_MISSING_SEGMENTS = {
    "resnet34_se": {"se"},
    "resnet50_se": {"se"},
    "resnet34_degconv_cpca_v3": {"cpca"},
    "resnet34_s2d_v1": {"stem_adapter"},
    "resnet34_msec_v1": {"msec"},
    "resnet34_msec_strips": {"msec"},
    "resnet34_msec_edge": {"msec"},
    "resnet34_msec_dil": {"msec"},
    "resnet34_msec_v1_p234": {"msec"},
    "swin_tiny_patch4_window7_224": {"conv_efg", "revert"},
    "resnet34_msec_dil_p234": {"gates"},
    "resnet34_msec_dil234": {"gates"},
    "resnet34_pfesa_dil": {"gates"},
    "resnet34_dil234w": {"gates"},
    "resnet34_dil2346": {"gates"},
    "resnet34_dil123": {"gates"},
    "resnet34_dil_r8": {"gates"},
    "resnet34_dil234_l4": {"gates"},
    "resnet34_dil234_l3": {"gates"},
    "resnet34_dil234_k5": {"gates"},
    "resnet34_dil234_sg": {"gates"},
    "resnet34_dil23_r": {"gates"},
    "resnet34_dil34_r": {"gates"},
    "resnet34_dil234_prog": {"gates"},
    "resnet34_dil234_mix": {"gates"},
    "resnet34_dil234_id": {"gates"},
    "resnet34_dil234_x2": {"gates"},
    "resnet34_atrous4_dil234": {"gates"},
    "resnet34_lib_haar": {"lib"},
    "resnet34_lib_msctx": {"lib"},
    "resnet34_lib_lcattn": {"lib"},
    "resnet34_dshortcut": {"downsample_avg"},
    "resnet34_lib_msctx_haar": {"lib"},
    "resnet34_lib_msctx_l4": {"lib"},
    "resnet34_lib_msctx_l3": {"lib"},
    "resnet34_hybrid_dil_msctx": {"lib"},
    # ConvNeXt-CLE 创新变体: 零门控新分支参数段(wds/wtc/wtg/msf*) 随机初始化属设计预期
    "convnext_wds": {"wds", "wds_gate"},
    "convnext_wtb": {"wtc", "wtg"},
    "convnext_msf": {"msf_p2", "msf_p3", "msf_g2", "msf_g3"},
    "convnext_wds_wtb": {"wds", "wds_gate", "wtc", "wtg"},
    "convnext_wds_wtb_msf": {"wds", "wds_gate", "wtc", "wtg", "msf_p2", "msf_p3", "msf_g2", "msf_g3"},
    "convnext_wds_wtb_l1": {"wds", "wds_gate", "wtc", "wtg"},
    "convnext_wds_wtb_s234": {"wds", "wds_gate", "wtc", "wtg"},
    "convnext_wds_wtb_k3": {"wds", "wds_gate", "wtc", "wtg"},
    "convnext_wtb_k3": {"wtc", "wtg"},
    "convnext_wds_wtb_k3_msf": {"wds", "wds_gate", "wtc", "wtg", "msf_p2", "msf_p3", "msf_g2", "msf_g3"},
    "convnext_wdd": {"wdd", "wdd_gate"},
    "convnext_wds_msf": {"wds", "wds_gate", "msf_p2", "msf_p3", "msf_g2", "msf_g3"},
    "convnext_wtb_msf": {"wtc", "wtg", "msf_p2", "msf_p3", "msf_g2", "msf_g3"},
    "convnext_wds_wdd_msf": {"wds", "wds_gate", "wdd", "wdd_gate", "msf_p2", "msf_p3", "msf_g2", "msf_g3"},
    # ConvNeXt-CLE V2 结构化变体
    "convnext_v2a": {"dwt", "conv"},
    "convnext_v2b": {"wavelet_convs", "wavelet_scale", "base_scale", "wt_filter", "iwt_filter"},
    "convnext_v2c": {"msf2_p2", "msf2_p3", "msf2_n2", "msf2_n3"},
    "convnext_v2ab": {"dwt", "conv", "wavelet_convs", "wavelet_scale", "base_scale", "wt_filter", "iwt_filter"},
    "convnext_v2abc": {"dwt", "conv", "wavelet_convs", "wavelet_scale", "base_scale", "wt_filter", "iwt_filter",
                       "msf2_p2", "msf2_p3", "msf2_n2", "msf2_n3"},
    "convnext_v2ac": {"dwt", "conv", "msf2_p2", "msf2_p3", "msf2_n2", "msf2_n3"},
    "convnext_v2bc": {"wavelet_convs", "wavelet_scale", "base_scale", "wt_filter", "iwt_filter",
                      "msf2_p2", "msf2_p3", "msf2_n2", "msf2_n3"},
    "convnext_v2bc4": {"wavelet_convs", "wavelet_scale", "base_scale", "wt_filter", "iwt_filter",
                       "msf2_p2", "msf2_p3", "msf2_n2", "msf2_n3"},
    # EPNet 三创新点变体 (convnext_ep.py, 2026-11-01): DPH 双池化头/FESR 频域重标定/LSDG 方向门控
    "convnext_dph": {"dph4_n", "dph3_a", "dph3_m", "dph3_an", "dph3_mn",
                     "dph2_a", "dph2_m", "dph2_an", "dph2_mn"},
    "convnext_fesr": {"fesr", "w_s", "w_e"},
    "convnext_lsdg": {"lsdg", "strip_h", "strip_v", "fuse"},
    "convnext_dph_fesr": {"dph4_n", "dph3_a", "dph3_m", "dph3_an", "dph3_mn",
                          "dph2_a", "dph2_m", "dph2_an", "dph2_mn", "fesr", "w_s", "w_e"},
    "convnext_dph_lsdg": {"dph4_n", "dph3_a", "dph3_m", "dph3_an", "dph3_mn",
                          "dph2_a", "dph2_m", "dph2_an", "dph2_mn", "lsdg", "strip_h",
                          "strip_v", "fuse"},
    "convnext_fesr_lsdg": {"fesr", "w_s", "w_e", "lsdg", "strip_h", "strip_v", "fuse"},
    "convnext_dph_fesr_lsdg": {"dph4_n", "dph3_a", "dph3_m", "dph3_an", "dph3_mn",
                               "dph2_a", "dph2_m", "dph2_an", "dph2_mn", "fesr", "w_s", "w_e",
                               "lsdg", "strip_h", "strip_v", "fuse"},
    "convnext_fesr2": {"fesr", "w_s", "w_e", "alpha"},
    "convnext_epnet": {"dph4_n", "dph3_a", "dph3_m", "dph3_an", "dph3_mn",
                       "dph2_a", "dph2_m", "dph2_an", "dph2_mn", "fesr", "w_s", "w_e", "alpha",
                       "lsdg", "strip_h", "strip_v", "fuse"},
    # dsd/fgc 的核均由投影初始化填充, 仅 dsm(混合权重)真正随机初始化
    "convnext_dsd": {"dsm"},
    "convnext_dph_dsd": {"dph4_n", "dph3_a", "dph3_m", "dph3_an", "dph3_mn",
                         "dph2_a", "dph2_m", "dph2_an", "dph2_mn", "dsm"},
    "convnext_dph_fgc": {"dph4_n", "dph3_a", "dph3_m", "dph3_an", "dph3_mn",
                         "dph2_a", "dph2_m", "dph2_an", "dph2_mn"},
    "convnext_fgc_dsd": {"dsm"},
    "convnext_epnet2": {"dph4_n", "dph3_a", "dph3_m", "dph3_an", "dph3_mn",
                        "dph2_a", "dph2_m", "dph2_an", "dph2_mn", "dsm"},
    "convnext_dph_fesr2": {"dph4_n", "dph3_a", "dph3_m", "dph3_an", "dph3_mn",
                           "dph2_a", "dph2_m", "dph2_an", "dph2_mn", "fesr", "w_s", "w_e", "alpha"},
    "convnext_dph_am": {"dph4_n", "dph3_a", "dph3_m", "dph3_an", "dph3_mn",
                         "dph2_a", "dph2_m", "dph2_an", "dph2_mn"},
"convnext_dph_mfm": {"dph4_n", "dph3_a", "dph3_m", "dph3_an", "dph3_mn",
                          "dph2_a", "dph2_m", "dph2_an", "dph2_mn", "mfm"},
    "convnext_dph_am2": {"dph4_n", "dph3_a", "dph3_m", "dph3_an", "dph3_mn",
                         "dph2_a", "dph2_m", "dph2_an", "dph2_mn", "am_aux"},
    "convnext_dph_mfm2": {"dph4_n", "dph3_a", "dph3_m", "dph3_an", "dph3_mn",
                          "dph2_a", "dph2_m", "dph2_an", "dph2_mn", "mfm"},
    "convnext_aep": {"dph4_n", "aep3", "aep2", "aep3_n", "aep2_n", "aep3_p", "aep2_p"},
}

# 各模型加载 ImageNet 预训练权重前需做的"权重变换"(如 s2d 变体的 conv1 通道膨胀)。
# 变换后仍需通过严格审计: 仅允许 fc 与白名单段缺失、零 unexpected。
def _inflate_conv1_for_s2d(state):
    """conv1 输入通道 3->12: 预训练核沿输入通道维复制 4 份并除以 4(保持输出量级)。"""
    w = state.get("conv1.weight")
    if w is not None and w.dim() == 4 and w.shape[1] == 3:
        state["conv1.weight"] = w.repeat(1, 4, 1, 1) / 4.0
    return state


def _remap_dshortcut(state):
    """ResNet-D 短接: checkpoint 的 downsample.0/.1 重映射为 conv/bn。"""
    out = {}
    for k, v in state.items():
        if ".downsample.0." in k:
            k = k.replace(".downsample.0.", ".downsample.conv.")
        elif ".downsample.1." in k:
            k = k.replace(".downsample.1.", ".downsample.bn.")
        out[k] = v
    return out


def _remap_stem2(state):
    """V2-A: 预训练 stem conv(3ch输入k4s4)与新 WaveletStem.conv(12ch输入k3s2)形状
    不兼容, 属设计预期 -> 删除; stem LayerNorm(形状相同)迁移为 WaveletStem.norm。"""
    out = {}
    for k, v in state.items():
        if k.startswith("downsample_layers.0.0."):
            continue  # 旧 stem conv 无法迁移, WaveletStem.conv 随机初始化
        if k.startswith("downsample_layers.0.1."):
            k = k.replace("downsample_layers.0.1.", "downsample_layers.0.norm.")
        out[k] = v
    return out


def _remap_wr(state):
    """V2-B: stage3/4 预训练 dwconv.{weight,bias} -> WTConv.base_conv.*。"""
    out = {}
    pat = re.compile(r"^(stages\.[23]\.[0-9]+)\.dwconv\.(weight|bias)$")
    for k, v in state.items():
        m = pat.match(k)
        k = f"{m.group(1)}.dwconv.base_conv.{m.group(2)}" if m else k
        out[k] = v
    return out


def _remap_stem2_wr(state):
    return _remap_wr(_remap_stem2(state))


def _remap_wr_s4(state):
    """仅 stage4(stages.3) 的 dwconv 重映射(v2bc4 只替换 stage4)。"""
    out = {}
    pat = re.compile(r"^(stages\.3\.[0-9]+)\.dwconv\.(weight|bias)$")
    for k, v in state.items():
        m = pat.match(k)
        k = f"{m.group(1)}.dwconv.base_conv.{m.group(2)}" if m else k
        out[k] = v
    return out


def _remap_dsd(state):
    """EPNet C-v2: stage2(stages.1) 的 7x7 dwconv 核 -> DSD 条带核投影初始化。
    (C,1,7,7) -> v(C,1,7,1)=Σ_w K, h(C,1,1,7)=Σ_h K (幅值守恒投影); bias -> dsv。"""
    out = {}
    for k, v in state.items():
        if k.startswith("stages.1.") and ".dwconv.weight" in k:
            w = v.float()
            base = k.replace(".dwconv.weight", ".dwconv.")
            out[base + "dsv.weight"] = w.sum(dim=3, keepdim=True)
            out[base + "dsh.weight"] = w.sum(dim=2, keepdim=True)
        elif k.startswith("stages.1.") and ".dwconv.bias" in k:
            out[k.replace(".dwconv.bias", ".dwconv.dsv.bias")] = v
        else:
            out[k] = v
    return out


def _remap_fgc(state):
    """EPNet B-v2: stage4(stages.3) 的 7x7 dwconv 核 -> 7x7 rfft2 频域滤波器初始值。
    核中心循环移位到原点后 rfft2 = 原 dwconv 的循环卷积频域对偶(投影初始化)。"""
    out = {}
    for k, v in state.items():
        if k.startswith("stages.3.") and ".dwconv.weight" in k:
            w = torch.roll(v.float().squeeze(1), shifts=(-3, -3), dims=(1, 2))  # 中心->原点
            out[k.replace(".dwconv.weight", ".dwconv.flt")] = torch.view_as_real(
                torch.fft.rfft2(w))                       # 7x7 canvas = 特征尺寸
        elif k.startswith("stages.3.") and ".dwconv.bias" in k:
            out[k.replace(".dwconv.bias", ".dwconv.flt_bias")] = v
        else:
            out[k] = v
    return out


MODEL_CHECKPOINT_TRANSFORMS = {
    "resnet34_s2d_v2": _inflate_conv1_for_s2d,
    "resnet34_dshortcut": _remap_dshortcut,
    # ConvNeXt-CLE V2 结构化变体的预训练键重映射
    "convnext_v2a": _remap_stem2,
    "convnext_v2b": _remap_wr,
    "convnext_v2ab": _remap_stem2_wr,
    "convnext_v2abc": _remap_stem2_wr,
    "convnext_v2ac": _remap_stem2,
    "convnext_v2bc": _remap_wr,
    "convnext_dsd": _remap_dsd,
    "convnext_fgc": _remap_fgc,
    "convnext_dph_dsd": _remap_dsd,
    "convnext_dph_fgc": _remap_fgc,
    "convnext_fgc_dsd": lambda st: _remap_fgc(_remap_dsd(st)),
    "convnext_epnet2": lambda st: _remap_fgc(_remap_dsd(st)),
    "convnext_v2bc4": _remap_wr_s4,
}
SWIN_VANILLA_IGNORED_CHECKPOINT_BUFFERS = {
    "layers.0.blocks.1.attn_mask",
    "layers.1.blocks.1.attn_mask",
    "layers.2.blocks.1.attn_mask",
    "layers.2.blocks.3.attn_mask",
    "layers.2.blocks.5.attn_mask",
}


# ───────── 进度条样式：与 evaluate_2class.py 的 rich 金色进度条一致 ─────────
# 进度条宽度 = 终端宽度 × 0.618；tqdm.rich 仍是 experimental，屏蔽其告警以保持输出整洁。
try:
    from tqdm import TqdmExperimentalWarning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
except Exception:
    pass
_TQDM_CONSOLE = Console(width=int(shutil.get_terminal_size().columns * 0.618))
_TQDM_OPTIONS = {'console': _TQDM_CONSOLE}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] seed = {seed} (set)")


def _git_commit(repo_dir):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def load_pretrained_weights_strict(model, weights_path, model_name):
    """Load pretrained weights and reject every non-classifier mismatch.

    The Microsoft Swin checkpoint stores five deterministic attention-mask
    buffers.  The local dynamic-resolution vanilla implementation recreates
    those masks at runtime instead of registering them in ``state_dict``.
    Only that exact frozen set is ignored, and only for the dedicated vanilla
    Swin registry key.  All trainable backbone parameters must still load.
    """
    path = Path(weights_path).resolve(strict=True)
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    if not isinstance(state, dict):
        raise TypeError(f"pretrained checkpoint is not a state_dict: {path}")
    state = dict(state)

    removed_classifier_keys = sorted(
        key for key in state if key.split(".", 1)[0] in CLASSIFIER_ROOTS
    )
    for key in removed_classifier_keys:
        del state[key]

    ckpt_transform = MODEL_CHECKPOINT_TRANSFORMS.get(model_name)
    if ckpt_transform is not None:
        state = ckpt_transform(state)

    ignored_buffer_keys = []
    if model_name in ("swin_tiny_vanilla_patch4_window7_224",
                      "swin_tiny_patch4_window7_224"):
        present_masks = {key for key in state if key.endswith(".attn_mask")}
        if present_masks != SWIN_VANILLA_IGNORED_CHECKPOINT_BUFFERS:
            raise RuntimeError(
                "vanilla Swin checkpoint attention-mask buffer set differs: "
                f"actual={sorted(present_masks)} "
                f"expected={sorted(SWIN_VANILLA_IGNORED_CHECKPOINT_BUFFERS)}"
            )
        ignored_buffer_keys = sorted(present_masks)
        for key in ignored_buffer_keys:
            del state[key]

    incompat = model.load_state_dict(state, strict=False)
    allowed_segments = MODEL_ALLOWED_MISSING_SEGMENTS.get(model_name, set())
    expected_missing = sorted(
        key for key in model.state_dict()
        if (key.split(".", 1)[0] in CLASSIFIER_ROOTS
            or any(seg in allowed_segments for seg in key.split(".")))
        # BN 的 num_batches_tracked 缓冲键: load_state_dict 不将其计入 missing_keys,
        # expected 侧同样排除, 保持两侧口径一致
        and not key.endswith("num_batches_tracked")
    )
    actual_missing = sorted(incompat.missing_keys)
    unexpected = sorted(incompat.unexpected_keys)
    if actual_missing != expected_missing or unexpected:
        raise RuntimeError(
            "pretrained backbone loading mismatch: "
            f"missing={actual_missing}, expected_missing={expected_missing}, "
            f"unexpected={unexpected}"
        )

    audit = {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "removed_classifier_keys": removed_classifier_keys,
        "ignored_nonparameter_buffer_keys": ignored_buffer_keys,
        "missing_keys": actual_missing,
        "unexpected_keys": unexpected,
        "allowed_missing_segments": sorted(allowed_segments),
        "checkpoint_transform": getattr(ckpt_transform, "__name__", None),
        "all_trainable_backbone_parameters_loaded": True,
    }
    print(
        f"[weights] strict load passed {path} | "
        f"classifier_missing={len(actual_missing)} "
        f"ignored_buffers={len(ignored_buffer_keys)} unexpected=0"
    )
    return audit


def prepare_training_samples(data_path_train, train_manifest="", data_path_val=None):
    """Run the historical Full scan, then optionally apply a validated manifest."""
    train_images_path, train_images_label = read_split_train_data(data_path_train)
    if not train_manifest:
        return train_images_path, train_images_label, None

    exclusion_roots = []
    if data_path_val:
        exclusion_roots.append(data_path_val)
        inferred_test = Path(data_path_val).resolve().parent / "test"
        if inferred_test.is_dir():
            exclusion_roots.append(inferred_test)
    train_images_path, train_images_label, manifest_info = filter_training_samples(
        train_images_path,
        train_images_label,
        data_path_train,
        train_manifest,
        exclusion_roots=exclusion_roots,
    )
    return train_images_path, train_images_label, manifest_info


def build_run_meta(
    args,
    img_size,
    norm_mean,
    norm_std,
    norm_source,
    save_path,
    manifest_info=None,
    pretrained_load_info=None,
):
    """Build legacy-compatible metadata, enriched only for manifest runs."""
    run_meta = {
        "model": args.model,
        "num_classes": args.num_classes,
        "img_size": img_size,
        "seed": args.seed,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "norm_source": norm_source,
        "selection_metric": "val_macro_f1",
        "loss": (f"label_smoothed_ce (smoothing={args.label_smoothing})"
                 if args.label_smoothing > 0 else "unweighted_cross_entropy"),
        "data_path_train": os.path.abspath(args.data_path_train),
        "data_path_val": os.path.abspath(args.data_path_val),
        "weights": os.path.abspath(save_path),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if pretrained_load_info is not None:
        run_meta["pretrained_load"] = pretrained_load_info
    if manifest_info is None:
        return run_meta

    repo_dir = Path(__file__).resolve().parent
    pretrained_path = Path(args.weights).resolve(strict=True) if args.weights else None
    run_meta.update({
        **manifest_info,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": 5e-2,
        "scheduler": f"CosineAnnealingLR(T_max={args.epochs}, eta_min=0)",
        "label_smoothing": args.label_smoothing,
        "early_stop_patience": args.early_stop,
        "amp_enabled": True,
        "best_checkpoint_metric": "val_macro_f1",
        "pretrained_weights": str(pretrained_path) if pretrained_path else None,
        "pretrained_sha256": sha256_file(pretrained_path) if pretrained_path else None,
        "train_root": str(Path(args.data_path_train).resolve()),
        "val_root": str(Path(args.data_path_val).resolve()),
        "git_commit": _git_commit(repo_dir),
        "code_hash": {
            "train_cle_fair.py": sha256_file(__file__),
            "consensus_manifest.py": sha256_file(repo_dir / "consensus_manifest.py"),
        },
        "timestamp": datetime.now().astimezone().isoformat(),
    })
    return run_meta


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, class_weights=None, label_smoothing=0.0, mfm_lambda=0.1):
    """单个 epoch 训练；进度条用与 evaluate_2class.py 一致的 rich 金色样式。

    训练逻辑与 utils.train_one_epoch 一致（CrossEntropy / AMP / AdamW），额外支持
    label_smoothing（默认 0 = 关闭，此时与原 UNWEIGHTED CE 行为完全等价）。仅把
    tqdm(file=sys.stdout, colour="magenta") 换成 tqdm.rich(options=_TQDM_OPTIONS)。
    """
    model.train()
    if class_weights is not None:
        loss_function = torch.nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)
    else:
        loss_function = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    use_amp = scaler is not None
    pbar = rich_tqdm(data_loader,
                     desc=f"train epoch {epoch} loss: 0.000, acc: 0.000",
                     options=_TQDM_OPTIONS)
    for step, data in enumerate(pbar):
        images, labels = data
        sample_num += images.shape[0]

        with torch.amp.autocast('cuda', enabled=use_amp):
            if getattr(model, "forward_labels", False):   # AMH: 目标类角度间隔需标签
                pred = model(images.to(device), labels.to(device))
            else:
                pred = model(images.to(device))
            loss = loss_function(pred, labels.to(device))
            mfm_mod = getattr(model, "mfm", None)          # MFM 自监督辅助损失聚合
            if mfm_mod is not None:
                loss = loss + getattr(model, "mfm_lambda", mfm_lambda) * mfm_mod.loss
            aux = getattr(model, "aux_logits", None)         # AMH 辅助 margin 正则
            if aux is not None:
                loss = loss + 0.3 * loss_function(aux, labels.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        scaler.scale(loss).backward() if use_amp else loss.backward()
        scaler.step(optimizer) if use_amp else optimizer.step()
        scaler.update() if use_amp else None
        optimizer.zero_grad()
        accu_loss += loss.detach()

        pbar.set_description("train epoch {} loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate_val_macro_f1(model, data_loader, device, label_smoothing=0.0):
    """Validation pass returning (val_loss, val_acc, val_macro_f1).
    CrossEntropy loss with the same label_smoothing as training, so train/val
    loss stay on the same scale (selection is by macro-F1, not loss)."""
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss, n_batches = 0.0, 0
    all_pred, all_true = [], []
    use_cuda = (device.type == "cuda")
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.amp.autocast('cuda', enabled=use_cuda):
            logits = model(images)
            loss = loss_fn(logits, labels)
        total_loss += loss.item()
        n_batches += 1
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_true.append(labels.cpu().numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    val_loss = total_loss / max(1, n_batches)
    val_acc = float(accuracy_score(y_true, y_pred))
    val_macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    return val_loss, val_acc, val_macro_f1


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── run dir (auto-suffix to avoid overwrite), named by model ──
    model_name = args.model
    os.makedirs(args.save_dir, exist_ok=True)
    base_dir = os.path.join(args.save_dir, model_name)
    if not os.path.exists(base_dir):
        run_dir = base_dir
    else:
        idx = 1
        while os.path.exists(f"{base_dir}-{idx}"):
            idx += 1
        run_dir = f"{base_dir}-{idx}"
    os.makedirs(run_dir, exist_ok=True)

    save_path = os.path.join(run_dir, args.save_name)
    csv_path = os.path.join(run_dir, args.csv_name)
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 4])

    train_images_path, train_images_label, manifest_info = prepare_training_samples(
        args.data_path_train,
        args.train_manifest,
        args.data_path_val,
    )
    val_images_path, val_images_label = read_split_val_data(args.data_path_val)
    if manifest_info is not None:
        print(
            "[manifest] "
            f"subset={manifest_info['subset']} "
            f"count={manifest_info['actual_train_count']} "
            f"normal={manifest_info['normal_count']} "
            f"benign={manifest_info['benign_count']} "
            f"malignant={manifest_info['malignant_count']} "
            f"sha256={manifest_info['manifest_sha256']}"
        )
    img_size = args.img_size

    # ── normalization: ImageNet if pretrained weights, else dataset-computed ──
    if args.weights:
        ds_mean = [0.485, 0.456, 0.406]
        ds_std = [0.229, 0.224, 0.225]
        norm_source = "imagenet"
        print("[norm] ImageNet 归一化(加载了预训练权重) "
              f"MEAN={ds_mean} STD={ds_std}")
    else:
        print("[norm] 数据集自算 mean/std(从零训练) ...")
        print(f"  transform: {_norm_transform_desc(img_size)}")
        ds_mean, ds_std, cache_status = get_or_compute_mean_std(
            args.data_path_train, train_images_path, img_size,
            cache_dir=args.norm_cache_dir,
            batch_size=args.batch_size, num_workers=nw,
            force_recompute=args.recompute_norm)
        print(f"  cache={cache_status} "
              f"MEAN=[{ds_mean[0]:.4f}, {ds_mean[1]:.4f}, {ds_mean[2]:.4f}] "
              f"STD=[{ds_std[0]:.4f}, {ds_std[1]:.4f}, {ds_std[2]:.4f}]")
        norm_source = "dataset_computed"

    data_transform = {
        "train": transforms.Compose([transforms.Resize((256, 256)),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.RandomRotation(degrees=15),
                                     transforms.RandomCrop(size=(img_size, img_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(ds_mean, ds_std)]),
        "val":   build_val_transform(img_size, ds_mean, ds_std)}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, pin_memory=True, num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, pin_memory=True, num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = build_model(args.model, num_classes=args.num_classes).to(device)

    # ── optional pretrained weights (kept from train.py, compact) ──
    pretrained_load_info = None
    if args.weights:
        pretrained_load_info = load_pretrained_weights_strict(
            model, args.weights, args.model
        )

    # ── save run metadata so eval can reuse the EXACT normalization ──
    run_meta = build_run_meta(
        args,
        img_size,
        ds_mean,
        ds_std,
        norm_source,
        save_path,
        manifest_info=manifest_info,
        pretrained_load_info=pretrained_load_info,
    )
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5e-2)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_macro_f1', 'lr'])

    print(f"[train] model={args.model} num_classes={args.num_classes} img={img_size} "
          f"bs={args.batch_size} epochs={args.epochs} | CE (label_smoothing={args.label_smoothing}) | select=val_macro_f1")
    print(f"[out]   run_dir={run_dir}")

    best_macro_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        cw = torch.tensor([float(x) for x in args.class_weights.split(",")]) \
            if getattr(args, "class_weights", "") else None
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer,
                                                 data_loader=train_loader, device=device,
                                                 epoch=epoch, scaler=scaler,
                                                 class_weights=cw,
                                                 label_smoothing=args.label_smoothing)

        val_loss, val_acc, val_macro_f1 = evaluate_val_macro_f1(model, val_loader, device,
                                                                label_smoothing=args.label_smoothing)

        scheduler.step()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch,
                                    f'{train_loss:.6f}', f'{train_acc:.6f}',
                                    f'{val_loss:.6f}', f'{val_acc:.6f}',
                                    f'{val_macro_f1:.6f}', f'{current_lr:.8f}'])

        # ── select best checkpoint by val MACRO-F1 (was val_acc in train.py) ──
        is_best = val_macro_f1 > best_macro_f1
        if is_best:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        flag = " *" if is_best else ""
        print(f"Epoch {epoch}: train_loss={train_loss:.3f}, train_acc={train_acc:.3f}, "
              f"val_loss={val_loss:.3f}, val_acc={val_acc:.3f}, val_macroF1={val_macro_f1:.4f}, "
              f"lr={current_lr:.6f}, best_macroF1={best_macro_f1:.4f} (epoch {best_epoch}){flag}")

        if args.early_stop > 0 and patience_counter >= args.early_stop:
            print(f"\n[early stop] epoch {epoch}: val_macro_f1 not improved for {args.early_stop} epochs")
            break

    total_time_str = str(timedelta(seconds=int(time.time() - start_time)))
    loss_desc = (f"CrossEntropy (label_smoothing={args.label_smoothing})"
                 if args.label_smoothing > 0 else "unweighted CrossEntropy")
    print()
    print("=" * 59)
    print(f"{'Training Summary (fair baseline)':^59}")
    print("=" * 59)
    print(f"  {'Model:':<20}{args.model}")
    print(f"  {'Best val macro-F1:':<20}{best_macro_f1:.4f}  (epoch {best_epoch})")
    print(f"  {'Selection:':<20}val macro-F1")
    print(f"  {'Loss:':<20}{loss_desc}")
    print(f"  {'Training Time:':<20}{total_time_str}")
    print(f"  {'Weights:':<20}{save_path}")
    print(f"  {'Run meta:':<20}{os.path.join(run_dir, 'run_meta.json')}")
    print("=" * 59)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (use 42/43/44 for repeats)')
    parser.add_argument('--model', type=str, default='fast_t0',
                        help='model name; one of: ' + ", ".join(sorted(MODEL_REGISTRY.keys())))
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='label smoothing for CrossEntropy loss (0 = off)')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--data-path-train', type=str, default="/path/to/cle/train",
                        help='patient-level train split directory')
    parser.add_argument('--train-manifest', type=str, default='',
                        help='optional CSV manifest filtering the already-scanned Full train set')
    parser.add_argument('--data-path-val', type=str, default="/path/to/cle/val",
                        help='patient-level val split directory')
    parser.add_argument('--weights', type=str, default='',
                        help='optional initial (pretrained) weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (e.g. cuda:0 or cpu)')
    parser.add_argument('--save-dir', type=str, default='./weights')
    parser.add_argument('--save-name', type=str, default='model_best.pth')
    parser.add_argument('--csv-name', type=str, default='train_log.csv')
    parser.add_argument('--early-stop', type=int, default=30,
                        help='early stopping patience on val_macro_f1 (0 = disabled)')
    parser.add_argument('--recompute-norm', action='store_true')
    parser.add_argument('--class-weights', type=str, default='',
                        help='逗号分隔各类CE权重(类序=sorted目录), 如 "1,1.5,1"')
    parser.add_argument('--norm-cache-dir', type=str, default='./norm_cache')
    opt = parser.parse_args()
    main(opt)
