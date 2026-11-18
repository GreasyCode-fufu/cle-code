import os
import argparse
import csv
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, cohen_kappa_score, matthews_corrcoef)

from models.Resnet import resnet34
from my_dataset import MyDataSet
from utils import read_split_val_data


MEAN = [0.2939, 0.2939, 0.2939]
STD = [0.2064, 0.2064, 0.2064]
CLASS_NAMES = ["abnormal", "normal"]  # class 0, class 1 (正类)


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def compute_binary_metrics(y_true, y_pred, y_prob):
    """计算二分类指标"""
    num_classes = 2
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    overall_acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    per_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    balanced_acc = np.mean(per_recall)

    # 正类 (class 1 = normal) 的二分类指标
    bin_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    bin_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    bin_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # AUROC / AUC-PR — 使用正类概率
    auroc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)

    # Weighted / Macro averages
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1_sklearn = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    paper_f1 = (2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
                if (weighted_precision + weighted_recall) > 0 else 0.0)

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

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
        "precision": bin_precision,
        "recall": bin_recall,
        "sensitivity": bin_recall,  # Sensitivity = Recall for positive class
        "specificity": per_class[CLASS_NAMES[1]]["specificity"],  # Specificity for positive class
        "f1": bin_f1,
        "auroc": auroc,
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
    """生成 Markdown 评估报告"""
    L = []
    L.append(f"# {method_name} 评估报告")
    L.append("")
    L.append(f"**正类**: `{CLASS_NAMES[1]}`  **负类**: `{CLASS_NAMES[0]}`  ")
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
        L.append(f"| 端到端推理时间/张 | {e2e:.4f} ms |")
        L.append("")

    # 核心指标
    L.append("## 1. 二分类核心指标")
    L.append("")
    L.append("| 指标 | 值 | 说明 |")
    L.append("|------|-----|------|")
    L.append(f"| Accuracy | {metrics['overall_accuracy']:.4f} | 正确预测数 / 总样本数 |")
    L.append(f"| Sensitivity / Recall | {metrics['recall']:.4f} | 正类 ({CLASS_NAMES[1]}) 召回率 |")
    L.append(f"| Specificity | {metrics['specificity']:.4f} | 负类 ({CLASS_NAMES[0]}) 召回率 |")
    L.append(f"| Precision | {metrics['precision']:.4f} | 正类 ({CLASS_NAMES[1]}) 精确率 |")
    L.append(f"| F1 Score | {metrics['f1']:.4f} | 2 * P * R / (P + R) |")
    L.append(f"| AUROC | {metrics['auroc']:.4f} | 正类 ({CLASS_NAMES[1]}) ROC 曲线下面积 |")
    L.append(f"| AUC-PR | {metrics['auc_pr']:.4f} | 正类 ({CLASS_NAMES[1]}) PR 曲线下面积 |")
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
    sep = "|---|------|------|"
    L.append(header)
    L.append(sep)
    for i, c_name in enumerate(CLASS_NAMES):
        row_vals = " | ".join([f"{cm[i, j]}" for j in range(2)])
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
    print(f"    Sensitivity:          {metrics['sensitivity']:.4f}  (正类 {CLASS_NAMES[1]})")
    print(f"    Specificity:          {metrics['specificity']:.4f}  (负类 {CLASS_NAMES[0]})")
    print(f"    Precision:            {metrics['precision']:.4f}  (正类 {CLASS_NAMES[1]})")
    print(f"    F1 Score:             {metrics['f1']:.4f}  (正类 {CLASS_NAMES[1]})")
    print(f"    AUROC:                {metrics['auroc']:.4f}")
    print(f"    AUC-PR:               {metrics['auc_pr']:.4f}")
    print(f"    Balanced Accuracy:    {metrics['balanced_accuracy']:.4f}")
    print(f"    Cohen's Kappa:        {metrics['kappa']:.4f}")
    print(f"    MCC:                  {metrics['mcc']:.4f}")
    if infer_per_img_ms is not None:
        print(f"    端到端推理时间(每张):   {infer_per_img_ms:.4f} ms")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── 输出根目录 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_dir, f"e2e_test_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    # ── 1. 加载模型 ──
    print("=" * 60)
    print("Step 1: 加载 ResNet34 模型")
    print("=" * 60)

    model = resnet34(num_classes=2)
    ckpt = torch.load(args.weights, map_location=device, weights_only=True)
    ckpt = {k: v for k, v in ckpt.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    print(f"  权重: {args.weights}")
    print(f"  输出目录: {base_dir}")
    print()

    # ── 2. 加载数据 ──
    print("Step 2: 加载测试集")
    transform = build_transform(args.img_size)

    test_paths, test_labels = read_split_val_data(args.data_test)
    test_dataset = MyDataSet(images_path=test_paths, images_class=test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=test_dataset.collate_fn)

    print(f"  测试集: {len(test_paths)} 张")
    print(f"  图片尺寸: {args.img_size}")
    print(f"  类别: {CLASS_NAMES[0]}(0), {CLASS_NAMES[1]}(1)")
    print()

    # ── 3. 端到端推理 ──
    print("Step 3: 端到端推理与评估")
    e2e_probs_list = []
    e2e_preds = []
    t0 = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            e2e_probs_list.append(probs.cpu().numpy())
            e2e_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    e2e_infer_time = time.time() - t0
    e2e_infer_per_img_ms = e2e_infer_time / len(test_paths) * 1000
    e2e_probs_2d = np.concatenate(e2e_probs_list, axis=0)
    e2e_preds = np.array(e2e_preds)
    test_labels_np = np.array(test_labels)

    # ── 4. 计算指标 ──
    metrics = compute_binary_metrics(test_labels_np, e2e_preds, e2e_probs_2d[:, 1])
    metrics["infer_time_total"] = e2e_infer_time
    metrics["infer_per_img_ms"] = e2e_infer_per_img_ms
    metrics["e2e_infer_per_img_ms"] = e2e_infer_per_img_ms

    # ── 5. 保存结果 ──
    save_predictions_csv(base_dir, test_paths, test_labels_np, e2e_preds, e2e_probs_2d, CLASS_NAMES)
    generate_markdown_report(metrics, "ResNet34-E2E", base_dir, test_labels_np, e2e_preds, e2e_probs_2d,
                             infer_time_total=e2e_infer_time, infer_per_img_ms=e2e_infer_per_img_ms,
                             feat_extract_per_img_ms=0.0)

    print(f"\n[saved] {base_dir}/predictions.csv")
    print(f"[saved] {base_dir}/evaluation_report.md")
    print()

    # ── 6. 打印结果摘要 ──
    print_metrics(metrics, "ResNet34-E2E", infer_per_img_ms=e2e_infer_per_img_ms)
    print()

    # ── 7. 打印汇总表 ──
    col_w = 20
    metric_keys = [
        ("overall_accuracy", "Accuracy"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score"),
        ("auroc", "AUROC"),
        ("auc_pr", "AUC-PR"),
        ("balanced_accuracy", "Balanced Acc"),
        ("kappa", "Cohen's Kappa"),
        ("mcc", "MCC"),
    ]
    table_w = 30 + col_w
    print("=" * table_w)
    print(f"{'ResNet34-E2E 评估结果':^{table_w}}")
    print("=" * table_w)
    print(f"{'指标':<30}{'值':>{col_w}}")
    print("-" * table_w)
    for key, label in metric_keys:
        val = metrics.get(key, 0.0)
        print(f"{label:<30}{val:>{col_w}.4f}")
    print("-" * table_w)
    print(f"{'推理时间/张 (ms)':<30}{e2e_infer_per_img_ms:>{col_w}.4f}")
    print(f"{'总推理时间 (s)':<30}{e2e_infer_time:>{col_w}.3f}")
    print("=" * table_w)
    print(f"\n所有结果已保存至: {base_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet34 端到端测试')
    parser.add_argument('--weights', type=str, required=True, help='训练好的 ResNet 权重路径')
    parser.add_argument('--data-test', type=str, required=True, help='测试集目录')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=str, default='./outputs')
    args = parser.parse_args()
    main(args)
