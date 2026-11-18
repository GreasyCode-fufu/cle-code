"""
二分类模型评估脚本
==================
支持通过命令行参数切换不同模型，对测试数据集进行全面的二分类评估。

使用方式:
  python evaluate_binary.py --model resnet34 --weights ./weights/best_model.pth --class-names positive negative
  python evaluate_binary.py --model repvit_m1 --weights ./weights/repvit_best.pth --img-size 224

评价指标（二分类核心指标）：
  - Accuracy: 总正确预测数 / 总样本数
  - Precision: 正类精确率
  - Recall / Sensitivity: 正类召回率
  - F1 Score: 2 * P * R / (P + R)
  - AUROC: ROC 曲线下面积（基于正类概率）
  - AUC-PR: PR 曲线下面积（基于正类概率）
  - Balanced Accuracy: 各类 Recall 的均值
  - Cohen's Kappa: 一致性系数
  - MCC: Matthews 相关系数

每类指标（One-vs-Rest）：
  - Precision, Recall, F1, Specificity, NPV

输出文件：
  1. 混淆矩阵图 (confusion_matrix.png)
  2. ROC 曲线图 (roc_curve.png)
  3. PR 曲线图 (pr_curve.png)
  4. 每类指标柱状图 (per_class_metrics.png)
  5. 预测 CSV (predictions.csv)
  6. Markdown 评估报告 (evaluation_report.md)
"""

import os
import sys
import csv
import time
import shutil
import argparse
import warnings
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.rich import tqdm
from rich.console import Console
from thop import profile, clever_format

from sklearn.metrics import (
    confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")

# 进度条宽度 = 终端宽度 × 0.618
_TQDM_CONSOLE = Console(width=int(shutil.get_terminal_size().columns * 0.618))
_TQDM_OPTIONS = {'console': _TQDM_CONSOLE}

# ========================== 模型注册表 ==========================
# 格式: "命令行名称": ("模块路径", "创建函数名")
MODEL_REGISTRY = {
    "resnet34":         ("models.Resnet", "resnet34"),
    "resnet50":         ("models.Resnet", "resnet50"),
    "resnet101":        ("models.Resnet", "resnet101"),
    "resnext50_32x4d":  ("models.Resnet", "resnext50_32x4d"),
    "resnext101_32x8d": ("models.Resnet", "resnext101_32x8d"),
    "repvit_m1":        ("models.RepViT", "repvit_m1"),
    "repvit_m2":        ("models.RepViT", "repvit_m2"),
    "repvit_m3":        ("models.RepViT", "repvit_m3"),
    "fast_t0":          ("models.FasterNet", "fast_t0"),
    "vanillanet_5":     ("models.VanilaNet", "vanillanet_5"),
    "alexnet":          ("models.AlexNet", "alexnet"),
    "convnext_tiny":    ("models.Convnext", "convnext_tiny"),
    "convnext_small":   ("models.Convnext", "convnext_small"),
    "efficientnet_b0":  ("models.EfficientNet", "efficientnet_b0"),
    "mobilenet_v2":     ("models.MobileNetV2", "mobilenet_v2"),
    "swin_tiny_patch4_window7_224": ("models.Swin", "swin_tiny_patch4_window7_224"),
    "vit_base_patch16_224":         ("models.Vit", "vit_base_patch16_224"),
}
# ===============================================================


# ───────────────────── 模型加载 ─────────────────────

def load_model(weights_path, device, model_name, num_classes):
    """根据模型名从注册表动态导入并加载权重"""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"未知模型: '{model_name}'，可用模型: {available}")

    module_name, func_name = MODEL_REGISTRY[model_name]
    create_model = getattr(__import__(module_name, fromlist=[func_name]), func_name)
    model = create_model(num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    valid_keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in valid_keys}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# ───────────────────── 数据加载与推理 ─────────────────────

def load_test_data(test_dir, class_names):
    """扫描测试集，返回 (图片路径列表, 标签数组)"""
    img_paths, labels = [], []
    supported = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    for idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"  [警告] 类别目录不存在: {cls_dir}")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(supported):
                img_paths.append(os.path.join(cls_dir, fname))
                labels.append(idx)
    return img_paths, np.array(labels)


def build_transform(img_size, norm_mean, norm_std):
    """与训练脚本中 val transform 一致"""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])


def run_inference(model, img_paths, transform, batch_size, device):
    """批量推理，返回概率矩阵 shape=(N, 2) 及推理效率指标"""
    use_cuda = device.type == "cuda"
    all_probs = []
    inference_time = 0.0
    total_samples = len(img_paths)

    end_to_end_start = time.time()

    for i in tqdm(range(0, len(img_paths), batch_size), desc="推理中", options=_TQDM_OPTIONS):
        batch_imgs = []
        for p in img_paths[i:i + batch_size]:
            img = Image.open(p).convert("RGB")
            batch_imgs.append(transform(img))
        batch_tensor = torch.stack(batch_imgs).to(device)

        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        if use_cuda:
            torch.cuda.synchronize()
        inference_time += time.time() - t0

        all_probs.append(probs)

    end_to_end_time = time.time() - end_to_end_start
    fps = total_samples / inference_time if inference_time > 0 else 0.0
    end_to_end_fps = total_samples / end_to_end_time if end_to_end_time > 0 else 0.0

    efficiency = {
        "inference_time": inference_time,
        "fps": fps,
        "end_to_end_time": end_to_end_time,
        "end_to_end_fps": end_to_end_fps,
        "total_samples": total_samples,
    }

    return np.concatenate(all_probs, axis=0), efficiency


# ───────────────────── 指标计算 ─────────────────────

def compute_binary_metrics(y_true, y_pred, y_prob, class_names):
    """计算所有二分类指标"""
    num_classes = 2
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # ── Overall metrics ──
    overall_acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    per_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    balanced_acc = np.mean(per_recall)

    # Binary metrics — 正类 (class 1) 为 positive
    bin_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    bin_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    bin_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # AUROC / AUC-PR — 使用正类概率
    auroc = roc_auc_score(y_true, y_prob[:, 1])
    auc_pr = average_precision_score(y_true, y_prob[:, 1])

    # Weighted / Macro averages
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1_sklearn = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    paper_f1 = (2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
                if (weighted_precision + weighted_recall) > 0 else 0.0)

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ── Per-class metrics (One-vs-Rest) ──
    per_class = {}
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        y_pred_c = (y_pred == c).astype(int)
        tp = int(np.sum((y_true_c == 1) & (y_pred_c == 1)))
        fp = int(np.sum((y_true_c == 0) & (y_pred_c == 1)))
        fn = int(np.sum((y_true_c == 1) & (y_pred_c == 0)))
        tn = int(np.sum((y_true_c == 0) & (y_pred_c == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Per-class AUC
        auc_roc_c = roc_auc_score(y_true_c, y_prob[:, c])
        auc_pr_c = average_precision_score(y_true_c, y_prob[:, c])

        per_class[class_names[c]] = {
            "precision": precision,
            "recall": recall,
            "sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
            "npv": npv,
            "auc_roc": auc_roc_c,
            "auc_pr": auc_pr_c,
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
        # Binary-specific
        "precision": bin_precision,
        "recall": bin_recall,
        "f1": bin_f1,
        "auroc": auroc,
        "auc_pr": auc_pr,
        # Averages
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1_sklearn": weighted_f1_sklearn,
        "paper_f1": paper_f1,
    }


# ───────────────────── 图表绘制 ─────────────────────

def plot_confusion_matrix(cm, class_names, save_dir):
    """绘制混淆矩阵"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n_cls = len(class_names)
    total = cm.sum()
    ax.set(xticks=np.arange(n_cls), yticks=np.arange(n_cls),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True label", xlabel="Predicted label",
           title="Confusion Matrix (Binary)")

    for i in range(n_cls):
        for j in range(n_cls):
            val = cm[i, j]
            pct = val / total * 100
            row_total = cm[i].sum()
            row_pct = val / row_total * 100 if row_total > 0 else 0
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)\n[{row_pct:.1f}%]",
                    ha="center", va="center", color=color, fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] confusion_matrix.png")


def plot_roc_curve(y_true, y_prob, class_names, save_dir):
    """绘制二分类 ROC 曲线"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))

    # 正类 ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc_val = roc_auc_score(y_true, y_prob[:, 1])
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2,
            label=f"{class_names[1]} (AUC={auc_val:.4f})")

    # 负类 ROC
    fpr_neg, tpr_neg, _ = roc_curve(1 - y_true, y_prob[:, 0])
    auc_neg = roc_auc_score(1 - y_true, y_prob[:, 0])
    ax.plot(fpr_neg, tpr_neg, color="#ff7f0e", linewidth=2,
            label=f"{class_names[0]} (AUC={auc_neg:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve (Binary)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=200)
    plt.close(fig)
    print("  [saved] roc_curve.png")


def plot_pr_curve(y_true, y_prob, class_names, save_dir):
    """绘制二分类 PR 曲线"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))

    # 正类 PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob[:, 1])
    ap = average_precision_score(y_true, y_prob[:, 1])
    ax.plot(rec, prec, color="#1f77b4", linewidth=2,
            label=f"{class_names[1]} (AP={ap:.4f})")

    # 负类 PR
    prec_neg, rec_neg, _ = precision_recall_curve(1 - y_true, y_prob[:, 0])
    ap_neg = average_precision_score(1 - y_true, y_prob[:, 0])
    ax.plot(rec_neg, prec_neg, color="#ff7f0e", linewidth=2,
            label=f"{class_names[0]} (AP={ap_neg:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve (Binary)", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "pr_curve.png"), dpi=200)
    plt.close(fig)
    print("  [saved] pr_curve.png")


def plot_per_class_metrics(metrics, class_names, save_dir):
    """绘制每类 Precision/Recall/F1/Specificity 柱状图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_keys = ["precision", "recall", "f1", "specificity"]
    metric_labels = ["Precision", "Recall (Sensitivity)", "F1-Score", "Specificity"]
    x = np.arange(len(class_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        vals = [metrics["per_class"][c][key] for c in class_names]
        bars = ax.bar(x + i * width, vals, width, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Metrics (Binary)", fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(class_names)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "per_class_metrics.png"), dpi=200)
    plt.close(fig)
    print("  [saved] per_class_metrics.png")


# ───────────────────── 报告生成 ─────────────────────

def generate_markdown_report(y_true, y_pred, y_prob, metrics,
                             model_name, weights_path, test_dir,
                             class_names, save_dir,
                             macs_str="", flops_str="", params_str="",
                             efficiency=None):
    """生成 Markdown 评估报告"""
    L = []
    L.append("# 二分类模型评估报告")
    L.append("")
    L.append(f"**模型**: `{model_name}`  ")
    L.append(f"**权重**: `{weights_path}`  ")
    L.append(f"**测试集**: `{test_dir}`  ")
    L.append(f"**类别数**: 2  ")
    L.append(f"**正类**: `{class_names[1]}`  **负类**: `{class_names[0]}`  ")
    L.append(f"**样本总数**: {len(y_true)}  ")
    for c_name in class_names:
        cnt = int((y_true == class_names.index(c_name)).sum())
        L.append(f"**{c_name}**: {cnt} 张  ")
    L.append("")

    # 1. 核心二分类指标
    L.append("## 1. 二分类核心指标")
    L.append("")
    L.append("| 指标 | 值 | 说明 |")
    L.append("|------|-----|------|")

    def _row(name, val, note=""):
        note_str = f" {note}" if note else ""
        L.append(f"| {name} | {val:.4f} |{note_str} |")

    _row("Accuracy", metrics["overall_accuracy"],
         "正确预测数 / 总样本数")
    _row("Precision", metrics["precision"],
         f"正类 ({class_names[1]}) 精确率")
    _row("Recall / Sensitivity", metrics["recall"],
         f"正类 ({class_names[1]}) 召回率")
    _row("F1 Score", metrics["f1"],
         "2 * P * R / (P + R)")
    _row("AUROC", metrics["auroc"],
         f"正类 ({class_names[1]}) ROC 曲线下面积")
    _row("AUC-PR", metrics["auc_pr"],
         f"正类 ({class_names[1]}) PR 曲线下面积")
    _row("Balanced Accuracy", metrics["balanced_accuracy"],
         "各类 Recall 均值")
    _row("Cohen's Kappa", metrics["kappa"],
         "一致性系数")
    _row("MCC", metrics["mcc"],
         "Matthews 相关系数")
    L.append("")

    # 2. 模型复杂度与推理效率
    L.append("## 2. 模型复杂度与推理效率")
    L.append("")
    L.append("### 模型复杂度")
    L.append("")
    L.append("| 指标 | 值 | 说明 |")
    L.append("|------|-----|------|")
    L.append(f"| MACs | {macs_str} | 乘加操作数 (thop profile) |")
    L.append(f"| FLOPs | {flops_str} | 1 MAC = 2 FLOPs |")
    L.append(f"| Params | {params_str} | 可学习参数总量 |")
    L.append("")

    if efficiency:
        L.append("### 推理速度")
        L.append("")
        L.append("| 指标 | 值 | 说明 |")
        L.append("|------|-----|------|")
        L.append(f"| Pure Inference FPS | {efficiency['fps']:.2f} | "
                 f"样本数 / 纯前向传播耗时 |")
        L.append(f"| Pure Inference Time | {efficiency['inference_time']:.3f}s | "
                 f"模型前向传播总耗时 |")
        L.append(f"| End-to-End FPS | {efficiency['end_to_end_fps']:.2f} | "
                 f"样本数 / 端到端总耗时 (含数据加载) |")
        L.append(f"| End-to-End Time | {efficiency['end_to_end_time']:.3f}s | "
                 f"含数据加载与预处理的端到端耗时 |")
        L.append(f"| Total Samples | {efficiency['total_samples']} | 测试集样本总数 |")
        L.append("")

    # 3. 补充总体指标
    L.append("## 3. 补充总体指标")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|------|-----|")

    def _row2(name, val):
        L.append(f"| {name} | {val:.4f} |")

    _row2("Macro Precision", metrics["macro_precision"])
    _row2("Macro Recall", metrics["macro_recall"])
    _row2("Macro F1-Score", metrics["macro_f1"])
    _row2("Weighted Precision", metrics["weighted_precision"])
    _row2("Weighted Recall", metrics["weighted_recall"])
    _row2("Weighted F1 (sklearn)", metrics["weighted_f1_sklearn"])
    _row2("Paper F1", metrics["paper_f1"])
    L.append("")

    # 4. 每类详细指标
    L.append("## 4. 每类详细指标（One-vs-Rest）")
    L.append("")
    for c_name in class_names:
        pc = metrics["per_class"][c_name]
        L.append(f"### {c_name} (support={pc['support']})")
        L.append("")
        L.append("| 指标 | 值 |")
        L.append("|------|-----|")

        def _prow(name, val):
            L.append(f"| {name} | {val:.4f} |")

        _prow("Precision (PPV)", pc["precision"])
        _prow("Recall (Sensitivity)", pc["recall"])
        _prow("Specificity", pc["specificity"])
        _prow("F1-Score", pc["f1"])
        _prow("NPV", pc["npv"])
        _prow("AUC-ROC", pc["auc_roc"])
        _prow("AUC-PR", pc["auc_pr"])
        L.append("")
        L.append(f"TP={pc['tp']}  FP={pc['fp']}  TN={pc['tn']}  FN={pc['fn']}")
        L.append("")

    # 5. 混淆矩阵
    L.append("## 5. 混淆矩阵")
    L.append("")
    cm = metrics["confusion_matrix"]
    header = "| | " + " | ".join([f"Pred: {c}" for c in class_names]) + " |"
    sep = "|---|------|------|"
    L.append(header)
    L.append(sep)
    for i, c_name in enumerate(class_names):
        row_vals = " | ".join([f"{cm[i, j]}" for j in range(2)])
        L.append(f"| True: {c_name} | {row_vals} |")
    L.append("")

    # 6. Classification Report (sklearn)
    L.append("## 6. Classification Report (sklearn)")
    L.append("")
    L.append("```")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    L.append(report)
    L.append("```")
    L.append("")

    report_path = os.path.join(save_dir, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  [saved] evaluation_report.md")
    return report_path


# ───────────────────── 主函数 ─────────────────────

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    class_names = args.class_names
    if len(class_names) != 2:
        raise ValueError(f"二分类脚本要求恰好 2 个类别名称，当前收到 {len(class_names)} 个: {class_names}")

    _run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./outputs/eval_binary_{args.model}_{_run_ts}"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 加载模型
    print(f"[1/6] 加载模型: {args.model}")
    print(f"  权重: {args.weights}")
    model = load_model(args.weights, device, args.model, num_classes=2)
    print(f"  设备: {device}")
    print(f"  二分类: {class_names[0]} (0) vs {class_names[1]} (1)")

    # 1.5 计算模型复杂度指标
    print(f"[2/6] 计算模型复杂度...")
    dummy_input = torch.zeros((1, 3, args.img_size, args.img_size)).to(device)
    macs_raw, params_raw = profile(model, inputs=(dummy_input,), verbose=False)
    flops_raw = 2 * macs_raw
    macs_str, params_str = clever_format([macs_raw, params_raw], "%.2f")
    flops_str, _ = clever_format([flops_raw, 0], "%.2f")
    print(f"  MACs:   {macs_str}")
    print(f"  FLOPs:  {flops_str}  (1 MAC = 2 FLOPs)")
    print(f"  Params: {params_str}")

    # 2. 加载测试数据
    print(f"[3/6] 加载测试数据: {args.data_path}")
    img_paths, y_true = load_test_data(args.data_path, class_names)
    for c_name in class_names:
        c_idx = class_names.index(c_name)
        print(f"  {c_name}: {np.sum(y_true == c_idx)} 张")
    print(f"  总计: {len(img_paths)} 张")

    if len(np.unique(y_true)) < 2:
        print("[错误] 测试集中只包含一个类别，无法进行二分类评估。")
        sys.exit(1)

    # 3. 推理
    print("[4/6] 推理中...")
    transform = build_transform(args.img_size, args.norm_mean, args.norm_std)
    y_prob, efficiency = run_inference(model, img_paths, transform, args.batch_size, device)
    y_pred = np.argmax(y_prob, axis=1)

    # 保存预测结果
    csv_path = os.path.join(save_dir, "predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        prob_headers = [f"prob_{c}" for c in class_names]
        writer.writerow(["image_path", "true_label", "pred_label"] + prob_headers)
        for i in range(len(y_true)):
            writer.writerow([img_paths[i], class_names[y_true[i]],
                             class_names[y_pred[i]]] + [f"{p:.6f}" for p in y_prob[i]])
    print(f"  [saved] predictions.csv")

    # 4. 计算指标
    print("[4/6] 计算评估指标...")
    metrics = compute_binary_metrics(y_true, y_pred, y_prob, class_names)

    # 5. 绘图 & 报告
    print("[5/6] 生成图表与报告...")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, save_dir)
    plot_roc_curve(y_true, y_prob, class_names, save_dir)
    plot_pr_curve(y_true, y_prob, class_names, save_dir)
    plot_per_class_metrics(metrics, class_names, save_dir)
    generate_markdown_report(y_true, y_pred, y_prob, metrics,
                             args.model, args.weights, args.data_path,
                             class_names, save_dir,
                             macs_str=macs_str, flops_str=flops_str,
                             params_str=params_str,
                             efficiency=efficiency)

    # 终端摘要
    print("\n" + "=" * 70)
    print(f"          二分类模型评估完成 — {args.model}")
    print("=" * 70)
    print(f"  Accuracy               : {metrics['overall_accuracy']:.4f}")
    print(f"  Precision              : {metrics['precision']:.4f}")
    print(f"  Recall / Sensitivity   : {metrics['recall']:.4f}")
    print(f"  F1 Score               : {metrics['f1']:.4f}")
    print(f"  AUROC                  : {metrics['auroc']:.4f}")
    print(f"  AUC-PR                 : {metrics['auc_pr']:.4f}")
    print(f"  Balanced Accuracy      : {metrics['balanced_accuracy']:.4f}")
    print(f"  Cohen's Kappa          : {metrics['kappa']:.4f}")
    print(f"  MCC                    : {metrics['mcc']:.4f}")
    print("-" * 70)
    print("  模型复杂度:")
    print(f"  MACs                   : {macs_str}")
    print(f"  FLOPs                  : {flops_str}  (1 MAC = 2 FLOPs)")
    print(f"  Params                 : {params_str}")
    print("-" * 70)
    print("  推理效率:")
    print(f"  Pure Inference FPS     : {efficiency['fps']:.2f}  "
          f"(总耗时 {efficiency['inference_time']:.3f}s)")
    print(f"  End-to-End FPS         : {efficiency['end_to_end_fps']:.2f}  "
          f"(总耗时 {efficiency['end_to_end_time']:.3f}s)")
    print(f"  Total Samples          : {efficiency['total_samples']}")
    print("-" * 70)

    for c_name in class_names:
        pc = metrics["per_class"][c_name]
        print(f"  [{c_name}]")
        print(f"    Precision={pc['precision']:.4f}  Recall={pc['recall']:.4f}  "
              f"F1={pc['f1']:.4f}  Specificity={pc['specificity']:.4f}  "
              f"AUC-ROC={pc['auc_roc']:.4f}  AUC-PR={pc['auc_pr']:.4f}")

    print("=" * 70)
    print(f"\n所有结果已保存至: {save_dir}")
    print(f"  - evaluation_report.md  (完整报告)")
    print(f"  - predictions.csv       (预测概率)")
    print(f"  - confusion_matrix.png  (混淆矩阵)")
    print(f"  - roc_curve.png         (ROC 曲线)")
    print(f"  - pr_curve.png          (PR 曲线)")
    print(f"  - per_class_metrics.png (每类指标柱状图)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="二分类模型评估脚本")

    # 模型相关
    #fast_t0
    parser.add_argument("--model", type=str, default="resnet34",
                        choices=sorted(MODEL_REGISTRY.keys()),
                        help="模型名称（需在 MODEL_REGISTRY 中注册）")
    parser.add_argument("--weights", type=str, default="/home/u2318483078/classifyML/weights/resnet34-2/resnet34_2class.pth",
                        help="模型权重文件路径")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="分类类别数（固定为 2）")


    # 数据相关
    parser.add_argument("--data-path", type=str, default="/home/u2318483078/2classCLEall/test",
                        help="测试集目录")
    parser.add_argument("--class-names", type=str, nargs="+",
                        default=["abnormal", "normal"],
                        help="类别名称列表（第 1 个为负类/0，第 2 个为正类/1，须与训练时一致）")
    parser.add_argument("--img-size", type=int, default=224,
                        help="输入图像尺寸")
    parser.add_argument("--norm-mean", type=float, nargs=3,
                        default=[0.2939, 0.2939, 0.2939],
                        help="归一化均值")
    parser.add_argument("--norm-std", type=float, nargs=3,
                        default=[0.2064, 0.2064, 0.2064],
                        help="归一化标准差")

    # 推理相关
    parser.add_argument("--batch-size", type=int, default=32,
                        help="推理批量大小")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="设备 (cuda:0 / cpu)")

    opt = parser.parse_args()
    if opt.num_classes != 2:
        print(f"[警告] --num-classes 设为 {opt.num_classes}，二分类脚本强制使用 2 个类别。")
        opt.num_classes = 2
    main(opt)
