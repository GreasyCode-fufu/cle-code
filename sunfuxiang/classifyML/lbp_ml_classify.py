import os
import argparse
import csv
import time
from datetime import datetime

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, cohen_kappa_score, matthews_corrcoef)
from joblib import Parallel, delayed

CLASS_NAMES = ["abnormal", "normal"]  # class 0, class 1 (正类)


# ───────────────────── LBP 特征配置 ─────────────────────

LBP_CONFIGS = {
    "LBP-10": {
        "desc": "1-scale: P=8,R=1, whole image",
        "dim": 10,
    },
    "LBP-54": {
        "desc": "3-scale: (8,1)+(16,2)+(24,3), whole image",
        "dim": 54,
    },
    "LBP-486": {
        "desc": "3x3 grid + 3-scale multi-scale",
        "dim": 486,
    },
}

SCALES = [(8, 1), (16, 2), (24, 3)]


# ───────────────────── 数据加载 ─────────────────────

def load_images_and_labels(root):
    """扫描目录，返回 (图片路径列表, 标签数组)，类别按字母序编号"""
    supported = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    class_dirs = sorted([d for d in os.listdir(root)
                         if os.path.isdir(os.path.join(root, d))])
    class_indices = {name: idx for idx, name in enumerate(class_dirs)}

    img_paths, labels = [], []
    for cls_name in class_dirs:
        cls_dir = os.path.join(root, cls_name)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(supported):
                img_paths.append(os.path.join(cls_dir, fname))
                labels.append(class_indices[cls_name])
    return img_paths, np.array(labels)


# ───────────────────── LBP 特征提取 ─────────────────────

def _lbp_hist(arr, P, R):
    """计算单块区域的 LBP uniform 直方图（L1 归一化）"""
    lbp = local_binary_pattern(arr, P=P, R=R, method='uniform')
    n_bins = P + 2
    hist = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1))[0].astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_lbp_10(gray_arr):
    """Config 1: P=8,R=1, 全图 → 10 维"""
    return _lbp_hist(gray_arr, 8, 1)


def extract_lbp_54(gray_arr):
    """Config 2: 多尺度全图 → 54 维"""
    hists = [_lbp_hist(gray_arr, P, R) for P, R in SCALES]
    return np.concatenate(hists)


def extract_lbp_486(gray_arr):
    """Config 3: 多尺度 3×3 分块 → 486 维 (≈ResNet 512)"""
    h, w = gray_arr.shape
    bh, bw = h // 3, w // 3
    hists = []
    for i in range(3):
        for j in range(3):
            block = gray_arr[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            for P, R in SCALES:
                hists.append(_lbp_hist(block, P, R))
    return np.concatenate(hists)


EXTRACT_FN = {
    "LBP-10": extract_lbp_10,
    "LBP-54": extract_lbp_54,
    "LBP-486": extract_lbp_486,
}


def _extract_single(img_path, img_size, config_name):
    """单张图：Resize → 灰度 → 按 config 提取特征"""
    img = Image.open(img_path).convert('L')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float64)
    return EXTRACT_FN[config_name](arr)


def extract_features_batch(img_paths, img_size, config_name, n_jobs=4):
    """并行批量提取特征"""
    features = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_extract_single)(p, img_size, config_name) for p in img_paths
    )
    return np.array(features)


# ───────────────────── 指标计算 ─────────────────────

def compute_binary_metrics(y_true, y_pred, y_prob):
    """与 extract_and_classify.py 完全一致"""
    num_classes = 2
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    overall_acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    per_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    balanced_acc = np.mean(per_recall)

    bin_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    bin_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    bin_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    auroc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)

    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1_sklearn = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    paper_f1 = (2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
                if (weighted_precision + weighted_recall) > 0 else 0.0)

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

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


# ───────────────────── 输出函数 ─────────────────────

def save_predictions_csv(save_dir, img_paths, y_true, y_pred, y_prob_2d):
    csv_path = os.path.join(save_dir, "predictions.csv")
    prob_headers = [f"prob_{c}" for c in CLASS_NAMES]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "pred_label"] + prob_headers)
        for i in range(len(y_true)):
            writer.writerow([img_paths[i], CLASS_NAMES[y_true[i]],
                             CLASS_NAMES[y_pred[i]]] + [f"{p:.6f}" for p in y_prob_2d[i]])
    return csv_path


def generate_markdown_report(metrics, method_name, config_name, save_dir, y_true,
                              train_time=None, infer_time_total=None, infer_per_image_ms=None):
    L = []
    L.append(f"# {method_name} 评估报告 ({config_name})")
    L.append("")
    L.append(f"**正类**: `{CLASS_NAMES[1]}`  **负类**: `{CLASS_NAMES[0]}`  ")
    L.append(f"**特征配置**: {config_name} — {LBP_CONFIGS[config_name]['desc']}  ")
    L.append(f"**特征维度**: {LBP_CONFIGS[config_name]['dim']}  ")
    L.append(f"**样本总数**: {len(y_true)}  ")
    for c_name in CLASS_NAMES:
        cnt = int((y_true == CLASS_NAMES.index(c_name)).sum())
        L.append(f"**{c_name}**: {cnt} 张  ")
    L.append("")

    if train_time is not None:
        L.append("## 0. 时间统计")
        L.append("")
        L.append("| 指标 | 数值 |")
        L.append("|------|------|")
        L.append(f"| 训练时间 | {train_time:.2f}s |")
        L.append(f"| 推理总时间 | {infer_time_total:.2f}s |")
        L.append(f"| 每张图推理时间 | {infer_per_image_ms:.3f} ms |")
        L.append(f"| LBP特征提取时间/张 | {metrics.get('feat_extract_per_img_ms', 0):.4f} ms |")
        L.append(f"| 端到端推理时间/张 | {metrics.get('e2e_infer_per_img_ms', 0):.4f} ms |")
        L.append("")

    L.append("## 1. 二分类核心指标")
    L.append("")
    L.append("| 指标 | 值 | 说明 |")
    L.append("|------|-----|------|")
    L.append(f"| Accuracy | {metrics['overall_accuracy']:.4f} | 正确预测数 / 总样本数 |")
    L.append(f"| Precision | {metrics['precision']:.4f} | 正类 ({CLASS_NAMES[1]}) 精确率 |")
    L.append(f"| Recall / Sensitivity | {metrics['recall']:.4f} | 正类 ({CLASS_NAMES[1]}) 召回率 |")
    L.append(f"| F1 Score | {metrics['f1']:.4f} | 2 * P * R / (P + R) |")
    L.append(f"| AUROC | {metrics['auroc']:.4f} | 正类 ({CLASS_NAMES[1]}) ROC 曲线下面积 |")
    L.append(f"| AUC-PR | {metrics['auc_pr']:.4f} | 正类 ({CLASS_NAMES[1]}) PR 曲线下面积 |")
    L.append(f"| Balanced Accuracy | {metrics['balanced_accuracy']:.4f} | 各类 Recall 均值 |")
    L.append(f"| Cohen's Kappa | {metrics['kappa']:.4f} | 一致性系数 |")
    L.append(f"| MCC | {metrics['mcc']:.4f} | Matthews 相关系数 |")
    L.append("")

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

    report_path = os.path.join(save_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return report_path


def print_metrics(metrics, label, train_time=None, infer_per_image_ms=None):
    print(f"\n  [{label}]")
    print(f"    Accuracy:             {metrics['overall_accuracy']:.4f}")
    print(f"    Precision:            {metrics['precision']:.4f}  (正类 {CLASS_NAMES[1]})")
    print(f"    Recall / Sensitivity: {metrics['recall']:.4f}  (正类 {CLASS_NAMES[1]})")
    print(f"    F1 Score:             {metrics['f1']:.4f}  (正类 {CLASS_NAMES[1]})")
    print(f"    AUROC:                {metrics['auroc']:.4f}")
    print(f"    AUC-PR:               {metrics['auc_pr']:.4f}")
    print(f"    Balanced Accuracy:    {metrics['balanced_accuracy']:.4f}")
    print(f"    Cohen's Kappa:        {metrics['kappa']:.4f}")
    print(f"    MCC:                  {metrics['mcc']:.4f}")
    if train_time is not None:
        print(f"    训练时间:             {train_time:.2f}s")
        print(f"    每张图推理时间:       {infer_per_image_ms:.3f} ms")
    if infer_per_image_ms is not None:
        print(f"    LBP特征提取时间(每张): {metrics.get('feat_extract_per_img_ms', 0):.4f} ms")
        print(f"    端到端推理时间(每张):  {metrics.get('e2e_infer_per_img_ms', 0):.4f} ms")


def run_method_pipeline(method_name, config_name, y_true, y_pred, y_prob_2d,
                        img_paths, base_dir,
                        train_time=None, infer_time_total=None, infer_per_image_ms=None,
                        feat_extract_per_img_ms=None, e2e_per_img_ms=None):
    metrics = compute_binary_metrics(y_true, y_pred, y_prob_2d[:, 1])
    metrics["feat_extract_per_img_ms"] = feat_extract_per_img_ms
    metrics["e2e_infer_per_img_ms"] = e2e_per_img_ms
    save_dir = os.path.join(base_dir, config_name, method_name)
    os.makedirs(save_dir, exist_ok=True)
    save_predictions_csv(save_dir, img_paths, y_true, y_pred, y_prob_2d)
    generate_markdown_report(metrics, method_name, config_name, save_dir, y_true,
                              train_time=train_time,
                              infer_time_total=infer_time_total,
                              infer_per_image_ms=infer_per_image_ms)
    print(f"  [saved] {save_dir}/")
    return metrics


# ───────────────────── 单个配置的完整实验 ─────────────────────

def run_config_experiment(config_name, train_paths, train_labels,
                          test_paths, test_labels, img_size, base_dir, n_jobs):
    """对一种 LBP 配置：提取特征 → 训练 3 种 ML → 评估"""
    config_dir = os.path.join(base_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)

    print(f"\n  提取 {config_name} 特征 ({LBP_CONFIGS[config_name]['desc']})...")
    t0 = time.time()
    train_feat = extract_features_batch(train_paths, img_size, config_name, n_jobs)
    train_extract_time = time.time() - t0

    t0 = time.time()
    test_feat = extract_features_batch(test_paths, img_size, config_name, n_jobs)
    test_extract_time = time.time() - t0

    print(f"  训练集: {train_feat.shape} ({train_extract_time:.1f}s)  "
          f"测试集: {test_feat.shape} ({test_extract_time:.1f}s)")

    test_extract_per_img_ms = test_extract_time / len(test_paths) * 1000

    np.save(os.path.join(config_dir, "train_features.npy"), train_feat)
    np.save(os.path.join(config_dir, "test_features.npy"), test_feat)

    scaler = StandardScaler()
    train_feat_s = scaler.fit_transform(train_feat)
    test_feat_s = scaler.transform(test_feat)

    results = {}

    # Logistic Regression
    print(f"\n  LogisticRegression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    t0 = time.time()
    lr.fit(train_feat_s, train_labels)
    lr_train_time = time.time() - t0

    t0 = time.time()
    lr_preds = lr.predict(test_feat_s)
    lr_probs_2d = lr.predict_proba(test_feat_s)
    lr_infer_time = time.time() - t0
    lr_infer_per_image_ms = lr_infer_time / len(test_paths) * 1000
    lr_e2e_per_img_ms = test_extract_per_img_ms + lr_infer_per_image_ms

    lr_metrics = run_method_pipeline(
        "LogisticRegression", config_name, test_labels,
        lr_preds, lr_probs_2d, test_paths, base_dir,
        train_time=lr_train_time, infer_time_total=lr_infer_time,
        infer_per_image_ms=lr_infer_per_image_ms,
        feat_extract_per_img_ms=test_extract_per_img_ms,
        e2e_per_img_ms=lr_e2e_per_img_ms)
    print_metrics(lr_metrics, f"{config_name}/LogReg",
                  train_time=lr_train_time, infer_per_image_ms=lr_infer_per_image_ms)
    results["LogReg"] = {**lr_metrics, "train_time": lr_train_time,
                          "infer_time_total": lr_infer_time,
                          "infer_per_image_ms": lr_infer_per_image_ms,
                          "feat_extract_per_img_ms": test_extract_per_img_ms,
                          "e2e_infer_per_img_ms": lr_e2e_per_img_ms}

    # SVM-Linear
    print(f"\n  SVM_Linear...")
    svm = SVC(kernel='linear', probability=True, random_state=42)
    t0 = time.time()
    svm.fit(train_feat_s, train_labels)
    svm_train_time = time.time() - t0

    t0 = time.time()
    svm_preds = svm.predict(test_feat_s)
    svm_probs_2d = svm.predict_proba(test_feat_s)
    svm_infer_time = time.time() - t0
    svm_infer_per_image_ms = svm_infer_time / len(test_paths) * 1000
    svm_e2e_per_img_ms = test_extract_per_img_ms + svm_infer_per_image_ms

    svm_metrics = run_method_pipeline(
        "SVM_Linear", config_name, test_labels,
        svm_preds, svm_probs_2d, test_paths, base_dir,
        train_time=svm_train_time, infer_time_total=svm_infer_time,
        infer_per_image_ms=svm_infer_per_image_ms,
        feat_extract_per_img_ms=test_extract_per_img_ms,
        e2e_per_img_ms=svm_e2e_per_img_ms)
    print_metrics(svm_metrics, f"{config_name}/SVM-Linear",
                  train_time=svm_train_time, infer_per_image_ms=svm_infer_per_image_ms)
    results["SVM-Linear"] = {**svm_metrics, "train_time": svm_train_time,
                              "infer_time_total": svm_infer_time,
                              "infer_per_image_ms": svm_infer_per_image_ms,
                              "feat_extract_per_img_ms": test_extract_per_img_ms,
                              "e2e_infer_per_img_ms": svm_e2e_per_img_ms}

    # Random Forest
    print(f"\n  RandomForest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    t0 = time.time()
    rf.fit(train_feat_s, train_labels)
    rf_train_time = time.time() - t0

    t0 = time.time()
    rf_preds = rf.predict(test_feat_s)
    rf_probs_2d = rf.predict_proba(test_feat_s)
    rf_infer_time = time.time() - t0
    rf_infer_per_image_ms = rf_infer_time / len(test_paths) * 1000
    rf_e2e_per_img_ms = test_extract_per_img_ms + rf_infer_per_image_ms

    rf_metrics = run_method_pipeline(
        "RandomForest", config_name, test_labels,
        rf_preds, rf_probs_2d, test_paths, base_dir,
        train_time=rf_train_time, infer_time_total=rf_infer_time,
        infer_per_image_ms=rf_infer_per_image_ms,
        feat_extract_per_img_ms=test_extract_per_img_ms,
        e2e_per_img_ms=rf_e2e_per_img_ms)
    print_metrics(rf_metrics, f"{config_name}/RF",
                  train_time=rf_train_time, infer_per_image_ms=rf_infer_per_image_ms)
    results["RF"] = {**rf_metrics, "train_time": rf_train_time,
                      "infer_time_total": rf_infer_time,
                      "infer_per_image_ms": rf_infer_per_image_ms,
                      "feat_extract_per_img_ms": test_extract_per_img_ms,
                      "e2e_infer_per_img_ms": rf_e2e_per_img_ms}

    return results


# ───────────────────── 主函数 ─────────────────────

def main(args):
    n_jobs = min(os.cpu_count(), 4)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_dir, f"lbp_ml_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    # ── 1. 加载数据 ──
    print("=" * 60)
    print("Step 1: 加载数据集")
    print("=" * 60)

    train_paths, train_labels = load_images_and_labels(args.data_train)
    test_paths, test_labels = load_images_and_labels(args.data_test)

    print(f"  训练集: {len(train_paths)} 张  ({args.data_train})")
    print(f"  测试集: {len(test_paths)} 张  ({args.data_test})")
    print(f"  类别: {CLASS_NAMES[0]}(0), {CLASS_NAMES[1]}(1)")
    print(f"  并行进程: {n_jobs}")
    print()

    # ── 2. 对每种 LBP 配置运行完整实验 ──
    print("=" * 60)
    print("Step 2: 逐配置提取 LBP 特征 + ML 分类")
    print("=" * 60)

    all_results = {}  # {config_name: {method_name: metrics}}
    for config_name in ["LBP-10", "LBP-54", "LBP-486"]:
        print(f"\n{'─' * 50}")
        print(f"  配置: {config_name}  ({LBP_CONFIGS[config_name]['desc']})")
        print(f"  维度: {LBP_CONFIGS[config_name]['dim']}")
        print(f"{'─' * 50}")
        all_results[config_name] = run_config_experiment(
            config_name, train_paths, train_labels,
            test_paths, test_labels, args.img_size, base_dir, n_jobs)

    # ── 3. 跨配置汇总对比 ──
    print("\n")
    print("=" * 90)
    print(f"{'Cross-Config Comparison (LBP Feature Dimensions)':^90}")
    print("=" * 90)

    # 构建汇总行：每个 (config, method) 组合
    combos = []
    for config_name in ["LBP-10", "LBP-54", "LBP-486"]:
        for ml_name in ["LogReg", "SVM-Linear", "RF"]:
            combos.append((config_name, ml_name))

    col_w = 14
    metric_keys = [
        ("overall_accuracy", "Accuracy"),
        ("auroc", "AUROC"),
        ("auc_pr", "AUC-PR"),
        ("f1", "F1 (pos)"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("balanced_accuracy", "Balanced Acc"),
        ("mcc", "MCC"),
        ("infer_per_image_ms", "Infer/img (ms)"),
        ("feat_extract_per_img_ms", "FeatExtract(ms)"),
        ("e2e_infer_per_img_ms",    "E2E/img (ms)"),
    ]

    # 按 ML 方法分组打印
    for ml_name in ["LogReg", "SVM-Linear", "RF"]:
        print(f"\n  [{ml_name}]")
        header = f"{'Metric':<20}" + "".join(f"{c[0]:>{col_w}}" for c in combos if c[1] == ml_name)
        print(header)
        print("-" * len(header))
        for key, label in metric_keys:
            row = f"{label:<20}"
            for config_name, m in combos:
                if m == ml_name:
                    row += f"{all_results[config_name][ml_name][key]:>{col_w}.4f}"
            print(row)

    print("\n" + "=" * 90)

    # ── 4. 保存汇总 CSV ──
    summary_path = os.path.join(base_dir, "all_configs_comparison.csv")
    csv_metric_keys = [
        ("overall_accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("auroc", "AUROC"),
        ("auc_pr", "AUC-PR"),
        ("balanced_accuracy", "Balanced_Accuracy"),
        ("kappa", "Kappa"),
        ("mcc", "MCC"),
        ("weighted_precision", "Weighted_Precision"),
        ("weighted_recall", "Weighted_Recall"),
        ("weighted_f1_sklearn", "Weighted_F1"),
        ("paper_f1", "Paper_F1"),
        ("macro_f1", "Macro_F1"),
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "dim", "method"] +
                        [k[1] for k in csv_metric_keys] +
                        ["Train_Time_s", "Infer_Total_s", "Infer_Per_Image_ms",
                         "FeatExtract_Per_Img_ms", "E2E_Infer_Per_Img_ms"])
        for config_name in ["LBP-10", "LBP-54", "LBP-486"]:
            dim = LBP_CONFIGS[config_name]["dim"]
            for ml_name in ["LogisticRegression", "SVM_Linear", "RandomForest"]:
                ml_key = ml_name.replace("LogisticRegression", "LogReg").replace("SVM_Linear", "SVM-Linear").replace("RandomForest", "RF")
                m = all_results[config_name][ml_key]
                writer.writerow([config_name, dim, ml_name] +
                                [f"{m[k]:.6f}" for k, _ in csv_metric_keys] +
                                [f"{m['train_time']:.4f}",
                                 f"{m['infer_time_total']:.4f}",
                                 f"{m['infer_per_image_ms']:.4f}",
                                 f"{m['feat_extract_per_img_ms']:.4f}",
                                 f"{m['e2e_infer_per_img_ms']:.4f}"])
    print(f"[saved] {summary_path}")

    print(f"\n所有结果已保存至: {base_dir}/")
    for config_name in ["LBP-10", "LBP-54", "LBP-486"]:
        print(f"  ├── {config_name}/")
        print(f"  │   ├── LogisticRegression/ (predictions.csv, report.md)")
        print(f"  │   ├── SVM_Linear/")
        print(f"  │   ├── RandomForest/")
        print(f"  │   ├── train_features.npy")
        print(f"  │   └── test_features.npy")
    print(f"  └── all_configs_comparison.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LBP 多配置特征提取 + ML 分类对比')
    parser.add_argument('--data-train', type=str, required=True, help='训练集目录')
    parser.add_argument('--data-test', type=str, required=True, help='测试集目录')
    parser.add_argument('--img-size', type=int, default=224, help='图像缩放尺寸')
    parser.add_argument('--output-dir', type=str, default='./lbp_ml_results', help='输出目录')
    args = parser.parse_args()
    main(args)
