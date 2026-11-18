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

import re
import torch
import numpy as np
import torch.optim as optim
from datetime import datetime, timedelta
from torchvision import transforms
from tqdm.rich import tqdm as rich_tqdm
from rich.console import Console

from my_dataset import MyDataSet

# HSD 辅助监督用的病理亚型词表(必须与 models/optimize/convnext_cle2.py 的
# SUBTYPE_NAMES / HSD_AGGREGATE 严格一致, 否则聚合矩阵语义会错位)
SUBTYPE_NAMES = ["atrophy", "intestinal_metaplasia", "normal", "dysplasia", "cancer"]
SUBTYPE_ID = {n: i for i, n in enumerate(SUBTYPE_NAMES)}


def _subtype_of(path):
    """从文件名解析病理亚型: train_atrophy_S_002e9e673681.jpg -> 'atrophy'。"""
    core = os.path.basename(path).split("_S_")[0]
    for p in ("train_", "val_", "test_"):
        core = core.replace(p, "")
    return SUBTYPE_ID.get(core, -1)
from utils import read_split_train_data, read_split_val_data
# Reuse train.py's dataset mean/std machinery verbatim (no modification to train.py):
from train_utils import get_or_compute_mean_std, _norm_transform_desc
from cle_fair_common import build_model, build_val_transform, MODEL_REGISTRY
from consensus_manifest import filter_training_samples, sha256_file

from sklearn.metrics import f1_score, accuracy_score


CLASSIFIER_ROOTS = {"head", "fc", "classifier"}

# 各模型"新增模块"允许缺失的参数段名：ImageNet 预训练权重中不存在这些结构，
# 随机初始化属于设计预期，不视为加载失败。段名与参数路径节点精确匹配。
MODEL_ALLOWED_MISSING_SEGMENTS = {
    "resnet34_se": {"se"},
    "densenet121": {"denselayer"},  # 旧名权重的 num_batches_tracked 等缓冲键
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
    # 外部创新网络 CLE TopoRel(models/toporel/): morphology 新分支段随机初始化属设计预期
    "convnext_tiny_cle_toporel": {"morphology"},
    "convnext_tiny_cle_toporel_relaux": {"morphology"},
    "convnext_tiny_cle_dose": ["dose"],
    "convnext_tiny_cle_dose_late": ["dose"],
    "convnext_tiny_cle_ptg": ["ptg1", "ptg2"],
    "convnext_tiny_cle_gds": ["gds"],
    "convnext_tiny_cle_dose_ptg": ["dose", "ptg1", "ptg2"],
    "convnext_tiny_cle_dose_late_ptg": ["dose", "ptg1", "ptg2"],
    "convnext_tiny_cle_dose_ptg_gds": ["dose", "gds", "ptg1", "ptg2"],
    "convnext_tiny_cle_dose_late_ptg_gds": ["dose", "gds", "ptg1", "ptg2"],
    "convnext_tiny_cle_ptg_gds": ["gds", "ptg1", "ptg2"],
    "convnext_tiny_cle_oca": ["oca1", "oca2"],
    "convnext_tiny_cle_oca_gds": ["gds", "oca1", "oca2"],
    "convnext_tiny_cle_oca_ptg_gds": ["gds", "oca1", "oca2", "ptg1", "ptg2"],
    "convnext_tiny_cle_dose_late_gds": ["dose", "gds"],
    "convnext_tiny_cle_oca_ptg": ["oca1", "oca2", "ptg1", "ptg2"],
    "convnext_tiny_cle_oca_wide": ["oca0", "oca1", "oca2"],
    "convnext_tiny_cle_oca_block": ["oca"],
    "convnext_tiny_cle_oca_strip": ["dose", "oca1", "oca2"],
    "convnext_v2bc": {"wavelet_convs", "wavelet_scale", "base_scale", "wt_filter", "iwt_filter", "msf2_p2", "msf2_p3", "msf2_n2", "msf2_n3"},
    "cle_convnext_sn": {"sn"},
    # CLE-Net 第二代: egb(无参数核心)/scg/hsc 新分支段随机初始化属设计预期
    "convnext_egb": {"egb", "egb_g"},
    "convnext_scg": {"scg", "scg_g"},
    "convnext_hsc": {"hsc"},
    "convnext_egb_scg": {"egb", "egb_g", "scg", "scg_g"},
    "convnext_egb_scg_msf": {"egb", "egb_g", "scg", "scg_g", "msf"},
    "convnext_scg_hsc": {"scg", "scg_g", "hsc"},
    # CLE-Net v2: ostem(重叠块茎)/norm_s3(双尺度头) 随机初始化属设计预期
    "convnext_os": {"ostem"},
    "convnext_dsh": {"norm_s3"},
    "convnext_os_scg": {"ostem", "scg", "scg_g"},
    "convnext_os_scg_dsh": {"ostem", "scg", "scg_g", "norm_s3"},
    # CLE-Net v3: pld(诊断原型亲和桥)随机初始化属设计预期
    "convnext_pld": {"pld"},
    "convnext_scg_pld": {"scg", "scg_g", "pld"},
    # S-Net v4: esa_g 标量门随机初始化属设计预期(esa 本体无参数)
    "convnext_esa": {"esa", "esa_g"},
    "convnext_esa_scg_hsc": {"esa", "esa_g", "scg", "scg_g", "hsc"},
    "convnext_egb_scg_hsc": {"egb", "egb_g", "scg", "scg_g", "hsc"},
    "convnext_egb_scg_hsd": {"egb", "egb_g", "scg", "scg_g", "hsd_sub", "hsd_A"},
    "convnext_egb_scg_hsdg": {"egb", "egb_g", "scg", "scg_g", "hsd_sub", "hsd_A", "hsd_g"},
    "convnext_egb_scg_fbe": {"egb", "egb_g", "scg", "scg_g", "fbe"},
    "convnext_egb_scg_fbe_all": {"egb", "egb_g", "scg", "scg_g", "fbe"},
    # 频域辅助正则化 SR: sr 段(proj 权重 + dct 常量缓冲)随机初始化属设计预期
    "convnext_egb_scg_sr": {"egb", "egb_g", "scg", "scg_g", "sr"},
    "convnext_sr": {"sr"},
    "convnext_egb_sr": {"egb", "egb_g", "sr"},
    "convnext_scg_sr": {"scg", "scg_g", "sr"},
    # S/B 底座 CLE-Net: 新分支段随机初始化属设计预期
    "convnext_small_egb_scg": {"egb", "egb_g", "scg", "scg_g"},
    "convnext_base_egb_scg": {"egb", "egb_g", "scg", "scg_g"},
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



def _remap_wr(state):
    """SF-ConvNeXt(v2bc): stage3/4 预训练 dwconv.{weight,bias} -> WTConv.base_conv.*。"""
    out = {}
    pat = re.compile(r"^(stages\.[23]\.[0-9]+)\.dwconv\.(weight|bias)$")
    for k, v in state.items():
        m = pat.match(k)
        k = f"{m.group(1)}.dwconv.base_conv.{m.group(2)}" if m else k
        out[k] = v
    return out

MODEL_CHECKPOINT_TRANSFORMS = {
    "convnext_v2bc": _remap_wr,
    # DenseNet 官方权重为旧键名(norm.1.weight), 重映射为模型当前键名(norm1.weight)
    "densenet121": lambda s: {
        (re.sub(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$",
                r"\1\2", k) if re.match(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$", k)
         else k): v
        for k, v in s.items()},
    "resnet34_s2d_v2": _inflate_conv1_for_s2d,
    "resnet34_dshortcut": _remap_dshortcut,
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
        "loss": (f"class_margin_ce (margin={args.class_margin})"
                 if args.class_margin > 0 else
                 (f"label_smoothed_ce (smoothing={args.label_smoothing})"
                  if args.label_smoothing > 0 else "unweighted_cross_entropy")),
        "data_path_train": os.path.abspath(args.data_path_train),
        "data_path_val": os.path.abspath(args.data_path_val),
        "weights": os.path.abspath(save_path),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if getattr(args, "sr_mode", "none") != "none" and getattr(args, "sr_weight", 0.0) > 0:
        run_meta["sr"] = {
            "mode": args.sr_mode,
            "weight": args.sr_weight,
            "margin": args.sr_margin,
            "latent_dim": 512,
            "bands": "low=25% mid=25% high=50%",
            "spr_order_if_spr": "Normal(2) < Benign(0) < Malignant(1)",
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


def _class_margin_ce(logits, targets, class_margin):
    """难类判别 margin CE：仅对 Benign(0)/Malignant(1) 的 target 位置施加
    固定 additive softmax margin m，Normal(2) margin=0。m=0 时严格等价 CE。"""
    m = torch.zeros(logits.size(1), device=logits.device, dtype=logits.dtype)
    m[0] = float(class_margin)   # Benign
    m[1] = float(class_margin)   # Malignant
    onehot = torch.nn.functional.one_hot(targets, logits.size(1)).to(logits.dtype)
    return torch.nn.functional.cross_entropy(logits - onehot * m.unsqueeze(0), targets)


_PAIR_PARTNER = [1, 0, 0]   # 各类的主要混淆伙伴: Benign<->Malignant, Normal->Benign


# ── 频域辅助正则化 SR (Spectral Regularisation, 见 experiments/SR_DESIGN.md) ──
# 类别索引: 0=Benign, 1=Malignant, 2=Normal(class_indices.json)。三类为无序病理
# 类型而非严格进展序列, 且 M3 checkpoint 实测 E_high 类别排序随种子/投影翻转
# (analysis/sr_baseline.json), 故主方案为无序判别式 CSR; spr 仅作对照验证臂。
_SR_SPR_ORDER = (2, 0, 1)   # spr 模式的严重度序: Normal(2) < Benign(0) < Malignant(1)


def _sr_loss(bands, targets, mode, margin):
    """频域辅助损失。

    bands: [B,3] = SRBranch 输出的三频带(low/mid/high)平均平方能量。
    csr(无序, 主方案): share = E/sum(E) 落单纯形且尺度不变(无幅值放大自由度);
        L = 类内紧致(share 到类心的平均平方欧氏距离)
            + 类间间隔(类心两两欧氏距离的 margin 铰链, 仅统计 batch 内真实
              出现的类别, 缺失类不参与也不置零)。
    spr(有序, 仅对照): 按 _SR_SPR_ORDER 的严重度序对 E_high 均值施加单调
        递增铰链(原始 SPR 公式)。
    返回 (loss, stats_dict)。
    """
    E = bands.float()
    y = targets
    if mode == "spr":
        mus = [E[y == c].mean(dim=0)[2] for c in _SR_SPR_ORDER if bool((y == c).any())]
        if len(mus) < 2:
            zero = E.sum() * 0.0
            return zero, {"spr": float(zero)}
        terms = [torch.relu(margin - (mus[i + 1] - mus[i])) for i in range(len(mus) - 1)]
        loss = torch.stack(terms).mean()
        return loss, {"spr": float(loss.detach())}

    s = E / (E.sum(dim=1, keepdim=True) + 1e-12)             # [B,3] 单纯形
    present = [c for c in range(3) if bool((y == c).any())]
    centers = {c: s[y == c].mean(dim=0) for c in present}
    lw = torch.stack([((s[y == c] - centers[c]) ** 2).sum(dim=1).mean() for c in present]).mean()
    pair_d = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            pair_d.append((centers[present[i]] - centers[present[j]]).pow(2).sum().sqrt())
    lb = (torch.stack([torch.relu(margin - d) for d in pair_d]).mean()
          if pair_d else lw * 0.0)
    stats = {"csr": float((lw + lb).detach()),
             "within": float(lw.detach()),
             "between": float(lb.detach()),
             "min_center_dist": float(torch.stack(pair_d).min().detach()) if pair_d else 0.0}
    return lw + lb, stats



def _pair_margin_term(logits, targets, pair_margin):
    """混淆伙伴定向间隔 (Confusion-Partner Margin, CPM) 的 pair 子项。

    对样本 (真实类 c) 取其混淆伙伴 p(c), 只在二元组 [z_c, z_p] 上做
    带间隔的二分类 CE:  CE([z_c - m, z_p], 0)。

    与"对真实类在**全类** softmax 上加间隔"的关键区别: CPM 完全不扰动
    真实类与非混淆类之间的几何关系(例如 Malignant 与 Normal 的边界原样保留)。
    动机: 观测显示 B<->M 互混是主要错误源, N->B 是次要但逐种子一致的来源(恒定 4 张);
    而统一 margin 会连带推坏本已干净的 M-N 边界(实测 seed44: M->N 由 1 张涨到 5 张,
    M 敏感度 0.9545 -> 0.9205)。m=0 时该子项退化为"混淆对二分类 CE"(纯加权, 无间隔)。
    """
    lut = torch.tensor(_PAIR_PARTNER, device=logits.device, dtype=torch.long)
    partner = lut[targets.long()]
    keep = partner != targets.long()          # 无混淆伙伴的类(如 Normal)完全不参与
    if not bool(keep.any()):
        return logits.sum() * 0.0
    lg = logits[keep]
    tg = targets.long()[keep]
    idx = torch.stack([tg, partner[keep]], dim=1)
    sub = torch.gather(lg, 1, idx)
    m = torch.zeros_like(sub)
    m[:, 0] = float(pair_margin)
    zero = torch.zeros(tg.size(0), dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(sub - m, zero)


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, class_weights=None, label_smoothing=0.0, class_margin=0.0, pair_margin=0.0, pair_weight=0.0, sub_aux_weight=0.0, sr_mode="none", sr_weight=0.0, sr_margin=0.15, sr_grad_check=False):
    """单个 epoch 训练；进度条用与 evaluate_2class.py 一致的 rich 金色样式。

    训练逻辑与 utils.train_one_epoch 一致（CrossEntropy / AMP / AdamW），额外支持
    label_smoothing（默认 0 = 关闭，此时与原 UNWEIGHTED CE 行为完全等价）、
    CPM pair margin 与 SR 频域辅助正则化（仅当 --sr-mode != none 且模型含
    SRBranch 时生效，经 model._sr_bands 读取频带能量）。仅把
    tqdm(file=sys.stdout, colour="magenta") 换成 tqdm.rich(options=_TQDM_OPTIONS)。
    """
    model.train()
    use_class_margin = class_margin > 0
    use_pair_margin = pair_weight > 0
    use_sr = sr_weight > 0 and sr_mode != "none"
    if use_sr and getattr(model, "use_sr", False) is False:
        raise RuntimeError(f"--sr-mode={sr_mode} 需要含 SRBranch 的模型(如 convnext_egb_scg_sr), "
                           f"当前 model={type(model).__name__} 不含该分支")
    if class_weights is not None:
        loss_function = torch.nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)
    else:
        loss_function = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    sr_stat_list = []
    optimizer.zero_grad()

    sample_num = 0
    use_amp = scaler is not None
    pbar = rich_tqdm(data_loader,
                     desc=f"train epoch {epoch} loss: 0.000, acc: 0.000",
                     options=_TQDM_OPTIONS)
    for step, data in enumerate(pbar):
        if len(data) == 3:
            images, labels, subs = data
        else:
            images, labels = data
            subs = None
        sample_num += images.shape[0]
        labels_dev = labels.to(device)
        sr_l = None

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred = model(images.to(device))
            if use_class_margin:
                loss_cls = _class_margin_ce(pred, labels_dev, class_margin)
            else:
                loss_cls = loss_function(pred, labels_dev)
            loss = loss_cls
            if use_pair_margin:
                loss = loss + pair_weight * _pair_margin_term(
                    pred, labels_dev, pair_margin)
            if sub_aux_weight > 0 and subs is not None:
                sub_logits = getattr(model, "_sub_logits", None)
                if sub_logits is not None:
                    keep = (subs >= 0).to(device)
                    if bool(keep.any()):
                        loss = loss + sub_aux_weight * torch.nn.functional.cross_entropy(
                            sub_logits[keep], subs.to(device)[keep])
            if use_sr:
                bands = getattr(model, "_sr_bands", None)
                if bands is not None:
                    sr_l, sr_stats = _sr_loss(bands, labels_dev, sr_mode, sr_margin)
                    loss = loss + sr_weight * sr_l
                    sr_stat_list.append(sr_stats)

        if sr_grad_check and use_sr and sr_l is not None and step == 0:
            # 一次性诊断: SR 梯度是否真正流入共享骨干/EGB/SCG, 以及与分类损失的量级比。
            # 两项同乘 1e4 再求梯度: 比值不变, 且避免初值 layer_scale=1e-6 的微小
            # 梯度在 fp16 下溢出为 0(与 GradScaler 的 loss 缩放同理)。
            named = list(model.named_parameters())
            bb = [(n, p) for n, p in named if p.requires_grad and not n.startswith("sr.")]
            srp = [(n, p) for n, p in named if p.requires_grad and n.startswith("sr.")]
            ps = [p for _, p in bb] + [p for _, p in srp]

            def _tot(gs):
                return sum(float(g.detach().float().pow(2).sum()) for g in gs if g is not None) ** 0.5
            g_cls = torch.autograd.grad(1e4 * loss_cls, ps, retain_graph=True, allow_unused=True)
            g_sr = torch.autograd.grad(1e4 * sr_weight * sr_l, ps, retain_graph=True, allow_unused=True)
            print(f"[sr-grad] ||dL_cls/dbackbone||={_tot(g_cls[:len(bb)]):.4f} "
                  f"||dL_sr/dbackbone||={_tot(g_sr[:len(bb)]):.4f} "
                  f"||dL_sr/dsr_proj||={_tot(g_sr[len(bb):]):.4f} "
                  f"ratio(bb)={_tot(g_sr[:len(bb)]) / max(_tot(g_cls[:len(bb)]), 1e-12):.4f}")

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels_dev).sum()

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

    sr_summary = None
    if sr_stat_list:
        keys = sr_stat_list[0].keys()
        sr_summary = {k: sum(d[k] for d in sr_stat_list) / len(sr_stat_list) for k in keys}
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, sr_summary


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
    nw = (args.num_workers if getattr(args, "num_workers", 0) > 0
          else min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 3]))

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

    if args.sub_aux_weight > 0:
        train_sub = [_subtype_of(p) for p in train_images_path]
        print("[hsd] subtype auxiliary supervision on (w=%g): " % args.sub_aux_weight
              + ", ".join("%s=%d" % (n, train_sub.count(i))
                          for i, n in enumerate(SUBTYPE_NAMES))
              + ", unknown=%d" % train_sub.count(-1))
    else:
        train_sub = None

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              images_subtype=train_sub,
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

    # ── optimizer param groups: isolate learnable gates from weight-decay, give them a
    #    higher LR so zero-initialised scalar gates can actually open. Root cause of the
    #    "zero-gate deadlock" found in GATE_DIAGNOSIS.md was the single-group AdamW with
    #    wd=5e-2 applied to gate scalars. ──
    GATE_SUFFIXES = ("egb_g", "scg_g", "esa_g", "hsd_g")
    gate_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(GATE_SUFFIXES) or n.endswith("hsc.g"):
            gate_params.append(p)
        else:
            other_params.append(p)
    pg = [
        {"params": other_params, "weight_decay": 5e-2},
        {"params": gate_params, "weight_decay": 0.0, "lr": args.lr * 100.0},
    ]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5e-2)
    print(f"[opt] param groups: backbone(n={len(other_params)}, lr={args.lr:g}, wd=5e-2) "
          f"| gates(n={len(gate_params)}, lr={args.lr*100.0:g}, wd=0)")
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_macro_f1', 'lr'])

    sr_desc = (f" | SR({args.sr_mode}, lambda={args.sr_weight:g}, margin={args.sr_margin:g})"
               if args.sr_mode != "none" and args.sr_weight > 0 else "")
    print(f"[train] model={args.model} num_classes={args.num_classes} img={img_size} "
          f"bs={args.batch_size} epochs={args.epochs} | CE (label_smoothing={args.label_smoothing}){sr_desc} | select=val_macro_f1")
    print(f"[out]   run_dir={run_dir}")

    best_macro_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    # ── 类别加权（处理5类不平衡）──
    if getattr(args, 'class_weight_mode', 'none') == 'none':
        class_weights = None
    else:
        _labels = np.asarray(train_images_label)
        _counts = np.bincount(_labels, minlength=args.num_classes).astype(np.float32)
        if args.class_weight_mode == 'inverse':
            _w = 1.0 / np.maximum(_counts, 1.0)
        else:
            _w = 1.0 / np.sqrt(np.maximum(_counts, 1.0))
        _w = _w / _w.sum() * args.num_classes
        class_weights = torch.tensor(_w, dtype=torch.float32)
        print('[class-weights] mode=%s counts=%s weights=%s' % (
            args.class_weight_mode, _counts.astype(int).tolist(),
            [round(x, 4) for x in _w.tolist()]))

    for epoch in range(args.epochs):
        train_loss, train_acc, sr_summary = train_one_epoch(
                                                 model=model, optimizer=optimizer,
                                                 data_loader=train_loader, device=device,
                                                 epoch=epoch, scaler=scaler,
                                                 class_weights=class_weights,
                                                 label_smoothing=args.label_smoothing,
                                                 class_margin=args.class_margin,
                                                 pair_margin=args.pair_margin,
                                                 pair_weight=args.pair_weight,
                                                 sub_aux_weight=args.sub_aux_weight,
                                                 sr_mode=args.sr_mode,
                                                 sr_weight=args.sr_weight,
                                                 sr_margin=args.sr_margin,
                                                 sr_grad_check=args.sr_grad_check)

        val_loss, val_acc, val_macro_f1 = evaluate_val_macro_f1(model, val_loader, device,
                                                                label_smoothing=args.label_smoothing)
        if sr_summary is not None:
            print(f"[sr] epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in sr_summary.items()))

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
    loss_desc = (f"Class-Margin CE (margin={args.class_margin})"
                 if args.class_margin > 0 else
                 (f"CrossEntropy (label_smoothing={args.label_smoothing})"
                  if args.label_smoothing > 0 else "unweighted CrossEntropy"))
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
    parser.add_argument('--class-weight-mode', type=str, default='none',
                        choices=['none', 'inverse', 'sqrt_inverse'],
                        help='class weighting for CE to handle imbalance (none/inverse/sqrt_inverse)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='label smoothing for CrossEntropy loss (0 = off)')
    parser.add_argument('--class-margin', type=float, default=0.0,
                        help='hard-class discriminative margin: additive softmax margin on Benign/Malignant (0 = off)')
    parser.add_argument('--pair-margin', type=float, default=0.0,
                        help='CPM: additive margin applied ONLY between true class and its confusion partner')
    parser.add_argument('--pair-weight', type=float, default=0.0,
                        help='CPM: weight of the pair-restricted CE term (0 = off)')
    parser.add_argument('--sub-aux-weight', type=float, default=0.0,
                        help='HSD: weight of the pathological-subtype auxiliary CE '
                             '(0 disables both the subtype labels and the auxiliary loss)')
    parser.add_argument('--sr-mode', type=str, default='none', choices=['none', 'csr', 'spr'],
                        help='SR 频域辅助正则化: none=关闭; csr=无序判别式(主方案); '
                             'spr=按严重度序 Normal<Benign<Malignant 的单调高频能量(仅对照)')
    parser.add_argument('--sr-weight', type=float, default=0.0,
                        help='SR: 辅助损失权重 λ (0 = off)')
    parser.add_argument('--sr-margin', type=float, default=0.15,
                        help='SR: 类心间隔 margin(share 单纯形上的欧氏距离尺度, '
                             '基线类心距实测 0.01~0.10, 上限 sqrt(2))')
    parser.add_argument('--sr-grad-check', action='store_true',
                        help='SR: 第一个 batch 打印一次梯度范数诊断(SR 梯度是否流入骨干/EGB/SCG)')
    parser.add_argument('--pair-partners', default='1,0,0',
                        help='CPM: confusion partner index per class (Benign,Malignant,Normal)')
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
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker 数; 0=自动 min(cpu, bs, 3)(历史默认), >0 显式指定")
    parser.add_argument('--norm-cache-dir', type=str, default='./norm_cache')
    opt = parser.parse_args()
    _PAIR_PARTNER[:] = [int(x) for x in opt.pair_partners.split(',')]
    main(opt)
