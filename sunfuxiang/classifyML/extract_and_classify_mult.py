# extract_and_classify_mult.py
# 本文件是 extract_and_classify.py 的多分类扩展版本。
# 将原有硬编码的二分类逻辑改为支持任意 N 分类任务，其余逻辑保持一致。
# 修改版：仅保留端到端模型测试，移除所有机器学习方法

import os
import argparse
import csv
import json
import hashlib
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, cohen_kappa_score, matthews_corrcoef)

from my_dataset import MyDataSet
from utils import read_split_train_data, read_split_val_data
from cle_fair_common import build_model, MODEL_REGISTRY


# ── 模型配置 ──
# 模型由 --model 选择（注册表见 cle_fair_common.MODEL_REGISTRY），与 train_cle_fair.py /
# evaluate_cle_fair.py 共用同一注册表与工厂函数，保证权重与架构严格匹配。
MODEL_NAME = "fast_t0"   # 占位；运行时由 args.model 覆盖（仅用于 argparse 描述/报告命名）


# ── 分类头移除策略 ──

def _detect_vanillanet(model):
    """VanillaNet 系列：cls1(含 Conv2d 分类头) + cls2 结构"""
    return hasattr(model, 'cls1') and hasattr(model, 'cls2')


def _remove_vanillanet_head(model, num_classes):
    for name, mod in model.named_modules():
        if (isinstance(mod, nn.Conv2d)
                and mod.kernel_size == (1, 1)
                and mod.out_channels == num_classes
                and mod.in_channels > num_classes):
            feat_dim = mod.in_channels
            model.cls1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
            model.cls2 = nn.Identity()
            return feat_dim
    raise ValueError("VanillaNet cls1 中未找到分类 Conv2d 层")


def _detect_deploy_vanillanet(model):
    """VanillaNet deploy 模式：cls(含 Conv2d 分类头)"""
    return (hasattr(model, 'cls')
            and not hasattr(model, 'cls1')
            and not hasattr(model, 'cls2')
            and any(isinstance(m, nn.Conv2d) for m in getattr(model, 'cls', []).modules()
                    if hasattr(getattr(model, 'cls'), 'modules')))


def _remove_deploy_vanillanet_head(model, num_classes):
    for mod in model.cls.modules():
        if (isinstance(mod, nn.Conv2d)
                and mod.kernel_size == (1, 1)
                and mod.out_channels == num_classes
                and mod.in_channels > num_classes):
            feat_dim = mod.in_channels
            model.cls = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
            return feat_dim
    raise ValueError("VanillaNet(deploy) cls 中未找到分类 Conv2d 层")


def _detect_last_linear(model):
    """通用策略：模型中存在 nn.Linear（ResNet/VGG/ViT/Swin/ConvNeXt/MKUNet 等）"""
    return any(isinstance(m, nn.Linear) for m in model.modules())


def _remove_last_linear_head(model, num_classes):
    last_name = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last_name = name
    if last_name is None:
        raise ValueError("模型中未找到 nn.Linear 层")
    parts = last_name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    attr = parts[-1]
    feat_dim = getattr(parent, attr).in_features
    setattr(parent, attr, nn.Identity())
    return feat_dim


def _detect_last_conv1x1(model):
    """兜底策略：模型中使用 Conv2d(1×1) 作为分类层"""
    return any(isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) for m in model.modules())


def _remove_last_conv1x1_head(model, num_classes):
    best_name, best_in_ch = None, 0
    for name, mod in model.named_modules():
        if (isinstance(mod, nn.Conv2d)
                and mod.kernel_size == (1, 1)
                and mod.out_channels == num_classes
                and mod.in_channels > best_in_ch):
            best_name = name
            best_in_ch = mod.in_channels
    if best_name is None:
        raise ValueError("模型中未找到 Conv2d(1×1) 分类层")
    top_attr = best_name.split('.')[0]
    setattr(model, top_attr, nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    ))
    for suffix in ['cls2', 'cls']:
        if hasattr(model, suffix) and suffix != top_attr:
            setattr(model, suffix, nn.Identity())
    return best_in_ch


# (检测函数, 移除函数, 策略名称, 适用模型描述)
_HEAD_STRATEGIES = [
    (_detect_vanillanet,        _remove_vanillanet_head,        "VanillaNet (cls1+cls2)",      "VanillaNet 系列"),
    (_detect_deploy_vanillanet, _remove_deploy_vanillanet_head, "VanillaNet deploy (cls)",      "VanillaNet deploy 模式"),
    (_detect_last_linear,       _remove_last_linear_head,       "nn.Linear → Identity",         "ResNet/VGG/ViT/Swin/ConvNeXt/MKUNet 等"),
    (_detect_last_conv1x1,      _remove_last_conv1x1_head,      "Conv2d(1×1) → Pool+Flatten",   "其他 Conv2d 分类头模型"),
]


def _prepare_feature_extractor(model, num_classes):
    """自动检测模型类型，移除分类头，返回 (feat_dim, 策略描述)。
    按优先级依次尝试 _HEAD_STRATEGIES 中的策略，命中即执行。"""
    for detect_fn, remove_fn, strat_name, _ in _HEAD_STRATEGIES:
        if detect_fn(model):
            feat_dim = remove_fn(model, num_classes)
            return feat_dim, strat_name
    raise ValueError(
        f"无法识别模型分类头结构。\n"
        f"  支持的策略:\n"
        + "\n".join(f"    - {desc}" for _, _, _, desc in _HEAD_STRATEGIES)
    )


# ── Mean/Std 自动计算（带 fingerprint 缓存）──

_NORM_CACHE_FILE = "dataset_norm_cache.json"


def _norm_transform_desc(img_size):
    resize_size = int(img_size * 1.143)
    crop_size = img_size
    return f"Resize({resize_size}) + CenterCrop({crop_size}) + ToTensor()"


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


def _resolve_norm(args, train_paths):
    """与 evaluate_cle_fair.py 公平评估协议对齐的归一化解析。

    优先级（高 → 低）：
      1. --norm-mean / --norm-std （CLI 显式覆盖，最高优先级）
      2. run_meta.json            （默认与 --weights 同目录，可用 --run-meta 指定）
      3. 重算训练集                （run_meta.json 不存在/无效时回退；--recompute-norm 强制走此路）

    直接读取训练时写入的 run_meta.json，保证测试归一化与训练完全一致。
    返回: (mean, std, source_str, cache_status_str)
    """
    # 1) 显式 CLI
    if args.norm_mean is not None and args.norm_std is not None:
        return list(args.norm_mean), list(args.norm_std), "cli (--norm-mean/--norm-std)", "n/a"

    # 2) run_meta.json
    meta_path = args.run_meta or os.path.join(
        os.path.dirname(os.path.abspath(args.weights)), "run_meta.json")
    if not args.recompute_norm and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "norm_mean" in meta and "norm_std" in meta:
                return list(meta["norm_mean"]), list(meta["norm_std"]), str(meta_path), "n/a"
            warnings.warn(f"run_meta.json 缺少 norm_mean/norm_std 字段，回退到重算: {meta_path}")
        except (json.JSONDecodeError, OSError) as e:
            warnings.warn(f"run_meta.json 读取失败 ({e})，回退到重算: {meta_path}")

    # 3) 重算训练集（带 fingerprint 缓存）
    if train_paths is None:
        raise ValueError(
            "无法确定归一化 mean/std：未提供 --norm-mean/--norm-std，"
            f"也未找到有效的 run_meta.json ({meta_path})，且未提供 --data-train 用于重算。"
            "请至少提供其中之一。")
    reason = "--recompute-norm" if args.recompute_norm else f"run_meta.json 未找到/无效 ({meta_path})"
    mean, std, cache_status = get_or_compute_mean_std(
        args.data_train, train_paths, args.img_size,
        cache_dir=args.norm_cache_dir,
        batch_size=args.batch_size, num_workers=4,
        force_recompute=args.recompute_norm)
    return mean, std, f"recompute ({reason})", cache_status

# CLASS_NAMES 在 main() 中动态设置
CLASS_NAMES = []


def auto_detect_class_names(data_dir):
    """从数据集目录自动检测类别名称（按子目录名排序）"""
    subdirs = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))])
    if not subdirs:
        raise ValueError(f"数据目录 {data_dir} 中未找到子目录（类别文件夹）")
    return subdirs


def build_transform(img_size, mean, std):
    resize_size = int(img_size * 1.143)
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            features = model(images.to(device))
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy() if isinstance(labels, np.ndarray) else labels.tolist())
    return np.concatenate(all_features, axis=0), np.array(all_labels)


def compute_multiclass_metrics(y_true, y_pred, y_prob, num_classes):
    """多分类指标计算（支持任意 N 分类）。

    Args:
        y_true: 真实标签 (n_samples,)
        y_pred: 预测标签 (n_samples,)
        y_prob: 完整概率矩阵 (n_samples, num_classes)
        num_classes: 类别数量
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    overall_acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    per_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    balanced_acc = np.mean(per_recall)

    # Macro-averaged precision / recall / f1
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1_val = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # AUROC — OvR macro
    auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    auroc_weighted = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')

    # AUC-PR — OvR 后取均值
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if y_true_bin.ndim == 1 or y_true_bin.shape[1] == 1:
        y_true_bin = np.column_stack([1 - y_true_bin.ravel(), y_true_bin.ravel()])
    auc_pr_scores = [average_precision_score(y_true_bin[:, c], y_prob[:, c])
                     for c in range(num_classes)]
    auc_pr = np.mean(auc_pr_scores)

    # Weighted averages
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1_sklearn = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    paper_f1 = (2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
                if (weighted_precision + weighted_recall) > 0 else 0.0)

    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class metrics (One-vs-Rest)
    per_class = {}
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        y_pred_c = (y_pred == c).astype(int)
        tp = int(np.sum((y_true_c == 1) & (y_pred_c == 1)))
        fp = int(np.sum((y_true_c == 0) & (y_pred_c == 1)))
        fn = int(np.sum((y_true_c == 1) & (y_pred_c == 0)))
        tn = int(np.sum((y_true_c == 0) & (y_pred_c == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        per_class[CLASS_NAMES[c]] = {
            "precision": prec, "recall": rec, "sensitivity": rec,
            "specificity": specificity, "f1": f1, "npv": npv,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "support": int(y_true_c.sum()),
        }

    return {
        "overall_accuracy": overall_acc,
        "balanced_accuracy": balanced_acc,
        "kappa": kappa,
        "mcc": mcc,
        "confusion_matrix": cm,
        "per_class": per_class,
        "precision": macro_prec,
        "recall": macro_rec,
        "f1": macro_f1_val,
        "auroc": auroc,
        "auroc_weighted": auroc_weighted,
        "auc_pr": auc_pr,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1_sklearn": weighted_f1_sklearn,
        "paper_f1": paper_f1,
    }


def save_predictions_csv(save_dir, img_paths, y_true, y_pred, y_prob, class_names):
    """保存 predictions.csv"""
    csv_path = os.path.join(save_dir, "predictions.csv")
    prob_headers = [f"prob_{c}" for c in class_names]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "pred_label"] + prob_headers)
        for i in range(len(y_true)):
            writer.writerow([img_paths[i], class_names[y_true[i]],
                             class_names[y_pred[i]]] + [f"{p:.6f}" for p in y_prob[i]])
    return csv_path


def generate_markdown_report(metrics, method_name, save_dir, y_true, y_pred, y_prob,
                              infer_time_total=None, infer_per_img_ms=None,
                              feat_extract_per_img_ms=0.0):
    """为单个方法生成 Markdown 评估报告"""
    num_classes = len(CLASS_NAMES)
    L = []
    L.append(f"# {method_name} 评估报告")
    L.append("")
    L.append(f"**类别数**: {num_classes}  ")
    L.append(f"**样本总数**: {len(y_true)}  ")
    for c_name in CLASS_NAMES:
        cnt = int((y_true == CLASS_NAMES.index(c_name)).sum())
        L.append(f"**{c_name}**: {cnt} 张  ")
    L.append("")

    if infer_time_total is not None:
        e2e = feat_extract_per_img_ms + (infer_per_img_ms or 0.0)
        L.append("## 0. 推理时间统计")
        L.append("")
        L.append("| 指标 | 数值 |")
        L.append("|------|------|")
        L.append(f"| 推理总时间 | {infer_time_total:.3f}s |")
        L.append(f"| 每张图推理时间 | {infer_per_img_ms:.3f} ms |")
        L.append(f"| 测试集图像总数 | {len(y_true)} 张 |")
        L.append(f"| {MODEL_NAME}特征提取时间/张 | {feat_extract_per_img_ms:.4f} ms |")
        L.append(f"| 端到端推理时间/张 | {e2e:.4f} ms |")
        L.append("")

    # 核心指标
    L.append("## 1. 多分类核心指标")
    L.append("")
    L.append("| 指标 | 值 | 说明 |")
    L.append("|------|-----|------|")
    L.append(f"| Accuracy | {metrics['overall_accuracy']:.4f} | 正确预测数 / 总样本数 |")
    L.append(f"| Precision | {metrics['precision']:.4f} | Macro-averaged 精确率 |")
    L.append(f"| Recall / Sensitivity | {metrics['recall']:.4f} | Macro-averaged 召回率 |")
    L.append(f"| F1 Score | {metrics['f1']:.4f} | Macro-averaged F1 |")
    L.append(f"| AUROC | {metrics['auroc']:.4f} | OvR Macro ROC 曲线下面积 |")
    L.append(f"| AUROC (weighted) | {metrics['auroc_weighted']:.4f} | OvR Weighted ROC 曲线下面积 |")
    L.append(f"| AUC-PR | {metrics['auc_pr']:.4f} | OvR Macro PR 曲线下面积均值 |")
    L.append(f"| Balanced Accuracy | {metrics['balanced_accuracy']:.4f} | 各类 Recall 均值 |")
    L.append(f"| Cohen's Kappa | {metrics['kappa']:.4f} | 一致性系数 |")
    L.append(f"| MCC | {metrics['mcc']:.4f} | Matthews 相关系数 |")
    L.append("")

    # 补充指标
    L.append("## 2. 补充总体指标")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|------|-----|")
    L.append(f"| Macro Precision | {metrics['macro_precision']:.4f} |")
    L.append(f"| Macro Recall | {metrics['macro_recall']:.4f} |")
    L.append(f"| Macro F1-Score | {metrics['macro_f1']:.4f} |")
    L.append(f"| Weighted Precision | {metrics['weighted_precision']:.4f} |")
    L.append(f"| Weighted Recall | {metrics['weighted_recall']:.4f} |")
    L.append(f"| Weighted F1 (sklearn) | {metrics['weighted_f1_sklearn']:.4f} |")
    L.append(f"| Paper F1 | {metrics['paper_f1']:.4f} |")
    L.append("")

    # 每类详细指标
    L.append("## 3. 每类详细指标（One-vs-Rest）")
    L.append("")
    for c_name in CLASS_NAMES:
        pc = metrics["per_class"][c_name]
        L.append(f"### {c_name} (support={pc['support']})")
        L.append("")
        L.append("| 指标 | 值 |")
        L.append("|------|-----|")
        L.append(f"| Precision (PPV) | {pc['precision']:.4f} |")
        L.append(f"| Recall (Sensitivity) | {pc['recall']:.4f} |")
        L.append(f"| Specificity | {pc['specificity']:.4f} |")
        L.append(f"| F1-Score | {pc['f1']:.4f} |")
        L.append(f"| NPV | {pc['npv']:.4f} |")
        L.append("")
        L.append(f"TP={pc['tp']}  FP={pc['fp']}  TN={pc['tn']}  FN={pc['fn']}")
        L.append("")

    # 混淆矩阵
    L.append("## 4. 混淆矩阵")
    L.append("")
    cm = metrics["confusion_matrix"]
    header = "| | " + " | ".join([f"Pred: {c}" for c in CLASS_NAMES]) + " |"
    sep = "|---|" + "|".join(["------" for _ in CLASS_NAMES]) + "|"
    L.append(header)
    L.append(sep)
    for i, c_name in enumerate(CLASS_NAMES):
        row_vals = " | ".join([f"{cm[i, j]}" for j in range(num_classes)])
        L.append(f"| True: {c_name} | {row_vals} |")
    L.append("")

    report_path = os.path.join(save_dir, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return report_path


def print_metrics(metrics, method_name, infer_per_img_ms=None):
    """终端打印单个方法的指标摘要"""
    print(f"\n  [{method_name}]")
    print(f"    Accuracy:             {metrics['overall_accuracy']:.4f}")
    print(f"    Precision:            {metrics['precision']:.4f}  (macro)")
    print(f"    Recall / Sensitivity: {metrics['recall']:.4f}  (macro)")
    print(f"    F1 Score:             {metrics['f1']:.4f}  (macro)")
    print(f"    AUROC:                {metrics['auroc']:.4f}")
    print(f"    AUROC (weighted):     {metrics['auroc_weighted']:.4f}")
    print(f"    AUC-PR:               {metrics['auc_pr']:.4f}")
    print(f"    Balanced Accuracy:    {metrics['balanced_accuracy']:.4f}")
    print(f"    Cohen's Kappa:        {metrics['kappa']:.4f}")
    print(f"    MCC:                  {metrics['mcc']:.4f}")
    if infer_per_img_ms is not None:
        print(f"    推理时间(每张):       {infer_per_img_ms:.3f} ms")
    if metrics.get('e2e_infer_per_img_ms') is not None:
        print(f"    端到端推理时间(每张):   {metrics['e2e_infer_per_img_ms']:.4f} ms")
    for c_name in CLASS_NAMES:
        pc = metrics["per_class"][c_name]
        print(f"    [{c_name}] Precision={pc['precision']:.4f}  Recall={pc['recall']:.4f}  "
              f"F1={pc['f1']:.4f}  Specificity={pc['specificity']:.4f}")


def run_method_pipeline(method_name, y_true, y_pred, y_prob_2d,
                        test_paths, base_dir, num_classes,
                        infer_time_total=None, infer_per_img_ms=None,
                        feat_extract_per_img_ms=0.0):
    """统一执行：计算指标 → 保存 CSV → 生成报告"""
    metrics = compute_multiclass_metrics(y_true, y_pred, y_prob_2d, num_classes)
    metrics["infer_time_total"] = infer_time_total
    metrics["infer_per_img_ms"] = infer_per_img_ms
    metrics["feat_extract_per_img_ms"] = feat_extract_per_img_ms
    metrics["e2e_infer_per_img_ms"] = feat_extract_per_img_ms + (infer_per_img_ms or 0.0)
    save_dir = os.path.join(base_dir, method_name)
    os.makedirs(save_dir, exist_ok=True)
    save_predictions_csv(save_dir, test_paths, y_true, y_pred, y_prob_2d, CLASS_NAMES)
    generate_markdown_report(metrics, method_name, save_dir, y_true, y_pred, y_prob_2d,
                              infer_time_total=infer_time_total, infer_per_img_ms=infer_per_img_ms,
                              feat_extract_per_img_ms=feat_extract_per_img_ms)
    print(f"  [saved] {save_dir}/predictions.csv, evaluation_report.md")
    return metrics


def main(args):
    global CLASS_NAMES, MODEL_NAME

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── 模型名校验 ──
    if args.model not in MODEL_REGISTRY:
        avail = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"未知 --model '{args.model}'。可选: {avail}")
    MODEL_NAME = args.model

    # ── 类别名称与数量 ──
    if args.class_names:
        CLASS_NAMES = list(args.class_names)
    else:
        CLASS_NAMES = auto_detect_class_names(args.data_train or args.data_test)
    num_classes = len(CLASS_NAMES)
    if args.num_classes is not None and args.num_classes != num_classes:
        raise ValueError(f"--num-classes={args.num_classes} 但检测到 {num_classes} 个类别: {CLASS_NAMES}")

    # ── 输出根目录 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_dir, f"ml_classify_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    # ── 1. 加载模型（去掉分类头，作为特征提取器）──
    print("=" * 60)
    print(f"Step 1: 加载 {MODEL_NAME} 特征提取器")
    print("=" * 60)

    model = build_model(args.model, num_classes=num_classes)
    feat_dim, head_strategy = _prepare_feature_extractor(model, num_classes=num_classes)
    ckpt = torch.load(args.weights, map_location=device, weights_only=True)
    ckpt = {k: v for k, v in ckpt.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    print(f"  权重: {args.weights}")
    print(f"  分类头策略: {head_strategy}")
    print(f"  特征维度: {feat_dim}")
    print(f"  类别数: {num_classes}")
    print(f"  输出目录: {base_dir}")
    print()

    # ── 2. 加载测试集 ──（端到端测试只需测试集；训练集仅用于 mean/std 重算兜底，可选）
    print("Step 2: 加载数据集")
    test_paths, test_labels = read_split_val_data(args.data_test)
    test_labels_np = np.array(test_labels)
    train_paths = None
    if args.data_train:
        train_paths, _ = read_split_train_data(args.data_train)
        print(f"  训练集: {len(train_paths)} 张  (仅用于 mean/std 重算兜底)")
    print(f"  测试集: {len(test_paths)} 张")

    # ── Mean/Std 解析（与 evaluate_cle_fair.py 公平评估协议对齐）──
    # 优先级：--norm-mean/--norm-std (CLI) > run_meta.json (与 --weights 同目录) > 重算训练集
    transform_desc = _norm_transform_desc(args.img_size)
    ds_mean, ds_std, norm_src, cache_status = _resolve_norm(args, train_paths)

    status_map = {
        "hit": "hit (缓存命中)",
        "miss": "miss (缓存未命中，已计算并写入)",
        "forced recompute": "forced recompute (--recompute-norm，已重算并覆盖)",
        "n/a": "n/a (来自 run_meta.json / CLI，未走重算路径)",
    }
    print(f"  Mean/Std source: {norm_src}")
    print(f"  img_size:        {args.img_size}")
    print(f"  Norm transform:  {transform_desc}")
    print(f"  Cache:           {status_map.get(cache_status, cache_status)}")
    print(f"  MEAN = [{ds_mean[0]:.4f}, {ds_mean[1]:.4f}, {ds_mean[2]:.4f}]")
    print(f"  STD  = [{ds_std[0]:.4f}, {ds_std[1]:.4f}, {ds_std[2]:.4f}]")
    print()

    transform = build_transform(args.img_size, ds_mean, ds_std)

    test_dataset = MyDataSet(images_path=test_paths, images_class=test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=test_dataset.collate_fn)

    print(f"  图片尺寸: {args.img_size}")
    print(f"  类别: {', '.join(f'{name}({i})' for i, name in enumerate(CLASS_NAMES))}")
    print()

    # ── 3. 端到端模型测试 ──
    e2e_dir_name = f"{MODEL_NAME}_E2E"
    print(f"Step 3: 端到端 {MODEL_NAME} 测试")
    print("=" * 60)

    model_e2e = build_model(args.model, num_classes=num_classes)
    model_e2e.load_state_dict(
        {k: v for k, v in torch.load(args.weights, map_location=device, weights_only=True).items()
         if k in model_e2e.state_dict() and v.shape == model_e2e.state_dict()[k].shape},
        strict=False)
    model_e2e.to(device)
    model_e2e.eval()

    e2e_probs_list = []
    e2e_preds = []
    t0 = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model_e2e(images.to(device))
            probs = torch.softmax(logits, dim=1)
            e2e_probs_list.append(probs.cpu().numpy())
            e2e_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    e2e_infer_time = time.time() - t0
    e2e_infer_per_img_ms = e2e_infer_time / len(test_paths) * 1000
    e2e_probs_2d = np.concatenate(e2e_probs_list, axis=0)
    e2e_preds = np.array(e2e_preds)

    e2e_metrics = run_method_pipeline(e2e_dir_name, test_labels_np, e2e_preds,
                                       e2e_probs_2d, test_paths, base_dir, num_classes,
                                       infer_time_total=e2e_infer_time,
                                       infer_per_img_ms=e2e_infer_per_img_ms,
                                       feat_extract_per_img_ms=0.0)
    print_metrics(e2e_metrics, f"{MODEL_NAME}-E2E", infer_per_img_ms=e2e_infer_per_img_ms)
    print()

    # ── 4. 保存测试特征（副产品，供下游分析；非评估必需）──
    print("Step 4: 保存测试特征")
    print("=" * 60)

    # 提取测试集特征
    test_features, _ = extract_features(model, test_loader, device)
    np.save(os.path.join(base_dir, "test_features.npy"), test_features)
    print(f"  测试集特征: {test_features.shape}")
    print(f"  [saved] test_features.npy")
    print()

    # ── 5. 完成 ──
    print("=" * 60)
    print(f"测试完成！所有结果已保存至: {base_dir}/")
    print("=" * 60)
    print(f"  ├── {e2e_dir_name}/")
    print(f"  │   ├── predictions.csv")
    print(f"  │   └── evaluation_report.md")
    print(f"  └── test_features.npy")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='端到端多分类测试（支持 cle_fair_common.MODEL_REGISTRY 中所有模型）')
    parser.add_argument('--model', type=str, default='fast_t0',
                        help='模型名（须与训练时一致）；可选: '
                             + ", ".join(sorted(MODEL_REGISTRY.keys())))
    parser.add_argument('--weights', type=str, required=True,
                        help='训练好的权重路径（须与 --model 匹配，如 ./weights_3class_seed42/<model>/<model>_3class_seed42_best.pth）')
    parser.add_argument('--data-train', type=str, default=None,
                        help='训练集目录（可选；仅在 run_meta.json 缺失/无效且需重算 mean/std 时使用）')
    parser.add_argument('--data-test', type=str, required=True, help='测试集目录')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--class-names', nargs='+', type=str, default=None,
                        help='类别名称列表（空格分隔），若未提供则从 --data-train 或 --data-test 目录自动检测')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='类别数量，若提供则用于验证（与自动检测或 --class-names 的数量须一致）')
    # Mean/Std 归一化参数（与 evaluate_cle_fair.py 公平评估协议对齐）
    parser.add_argument('--norm-mean', type=float, nargs=3, default=None,
                        help='手动覆盖归一化均值（最高优先级，覆盖 run_meta.json 与重算）')
    parser.add_argument('--norm-std', type=float, nargs=3, default=None,
                        help='手动覆盖归一化标准差（最高优先级，覆盖 run_meta.json 与重算）')
    parser.add_argument('--run-meta', type=str, default=None,
                        help='run_meta.json 路径（默认: --weights 同目录下的 run_meta.json）')
    # Mean/Std 缓存参数
    parser.add_argument('--recompute-norm', action='store_true',
                        help='强制重新计算 mean/std，忽略 run_meta.json 与已有缓存')
    parser.add_argument('--norm-cache-dir', type=str, default='./norm_cache',
                        help='mean/std 缓存目录（仅当走重算路径时使用）')
    args = parser.parse_args()
    main(args)
