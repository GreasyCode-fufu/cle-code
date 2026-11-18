# -*- coding: utf-8 -*-
"""
extract_and_classify_dual.py — 双流融合模型的特征提取 + ML 分类 (classifyML/bow_fusion 版)

流程沿用 ../../BoW/extract_and_classify_dual.py (双流特征提取 + 多个 ML 分类器对比)，
但**评价指标完全复用 classifyML/evaluate.py 的函数**，保证与单流评估口径一致:
  - evaluate.compute_metrics(...)   主指标 macro-F1 / Accuracy / AUROC(OvR) / AUC-PR /
                                    Kappa / MCC / Balanced-Acc + 每类 Sens/Spec/Precision/F1/NPV
  - evaluate.bootstrap_ci(...)      核心指标 95% Bootstrap CI (--bootstrap 开启)
  - evaluate.print_report(...)      终端规范报告
  - evaluate.save_outputs(...)      predictions.csv / results.json / evaluation_report.md
classifyML 原始文件 (evaluate.py 等) 未被修改。

双流特有逻辑 (与 BoW 版一致):
  - 模型为 dual_stream，forward 需要 (images, bow_feat) 双输入
  - 数据集 MyDataSetBOW，需预提取 LBP-BoW 特征
  - 特征提取 / E2E 推理循环均传入 bow_feat
  - 分类头移除: classifier → Identity，特征维度 = 2*d_model (=classifier[0].in_features)

用法 (从 classifyML 根目录运行):
  python bow_fusion/extract_and_classify_dual.py \
      --weights bow_fusion/weights/dual_fast_t0/model_best.pth \
      --data-train /path/cle/train --data-test /path/cle/test --img_size 224
"""

import os
import sys
import csv
import json
import time
import hashlib
import argparse
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── 确保 classifyML 根目录在 sys.path，以便 import cle_fair_common / utils / evaluate ──
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils import read_split_train_data, read_split_val_data
from cle_fair_common import build_val_transform
import evaluate  # 复用 evaluate.py 的指标函数

from bow_extractor import extract_and_cache_bow, BOWFeatureCache
from my_dataset_bow import MyDataSetBOW
from dual_stream_model import dual_stream as create_model

warnings.filterwarnings("ignore")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── 归一化解析 (与 evaluate.resolve_norm 同口径) ──

def resolve_norm(weights, norm_mean, norm_std, run_meta=None):
    if norm_mean is not None and norm_std is not None:
        return list(norm_mean), list(norm_std), "cli"
    meta_path = run_meta or os.path.join(
        os.path.dirname(os.path.abspath(weights)), "run_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "norm_mean" in meta and "norm_std" in meta:
            return meta["norm_mean"], meta["norm_std"], f"run_meta({meta.get('norm_source', '?')})"
    print("  [警告] 未找到 run_meta.json 也未传 --norm-mean/std，回退 ImageNet。")
    return IMAGENET_MEAN, IMAGENET_STD, "imagenet_fallback"


def auto_detect_class_names(data_dir):
    subdirs = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))])
    if not subdirs:
        raise ValueError(f"数据目录 {data_dir} 中未找到子目录（类别文件夹）")
    return subdirs


def compute_dual_complexity(model, img_size, bow_dim, device):
    """双输入复杂度 (thop)。返回与 evaluate.compute_complexity 相同结构的 dict。"""
    try:
        from thop import profile, clever_format
        dummy_img = torch.zeros((1, 3, img_size, img_size)).to(device)
        dummy_bow = torch.zeros((1, bow_dim)).to(device)
        macs_raw, params_raw = profile(model, inputs=(dummy_img, dummy_bow), verbose=False)
        flops_raw = 2 * macs_raw
        macs_str, params_str = clever_format([macs_raw, params_raw], "%.2f")
        flops_str, _ = clever_format([flops_raw, 0], "%.2f")
        for mod in model.modules():
            for attr in ("total_ops", "total_params"):
                if hasattr(mod, attr):
                    delattr(mod, attr)
        return {"macs": macs_str, "flops": flops_str, "params": params_str,
                "params_raw": int(params_raw), "macs_raw": int(macs_raw)}
    except Exception as e:
        print(f"  [提示] 复杂度计算跳过 (thop 不可用): {e}")
        return {"macs": "N/A", "flops": "N/A", "params": "N/A", "params_raw": 0, "macs_raw": 0}


def extract_features(model, dataloader, device):
    """提取特征 — 适配 (images, bow_feats, labels) 三元组 (classifier 已置 Identity)。"""
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for images, bow_feats, labels in dataloader:
            features = model(images.to(device), bow_feats.to(device))
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy() if isinstance(labels, np.ndarray) else labels.tolist())
    return np.concatenate(all_features, axis=0), np.array(all_labels)


def make_eff(pure_time, e2e_time, n):
    """构造 evaluate.print_report / save_outputs 需要的 efficiency dict。"""
    return {
        "pure_infer_time": pure_time,
        "pure_fps": n / pure_time if pure_time > 0 else 0.0,
        "pure_ms_per_img": pure_time / n * 1000 if n else 0.0,
        "e2e_time": e2e_time,
        "e2e_fps": n / e2e_time if e2e_time > 0 else 0.0,
        "total_samples": n,
    }


def run_method(method_label, dir_name, y_true, y_pred, y_prob, test_paths,
               base_dir, num_classes, class_names, complexity, eff,
               weights_path, test_dir, img_size, norm_info, args, do_print=True):
    """单方法流水线: compute_metrics (+bootstrap) → save_outputs → 紧凑打印。"""
    m = evaluate.compute_metrics(y_true, y_pred, y_prob, num_classes, class_names)
    ci = None
    if args.bootstrap:
        ci = evaluate.bootstrap_ci(y_true, y_prob, num_classes,
                                   n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed)
    save_dir = os.path.join(base_dir, dir_name)
    evaluate.save_outputs(save_dir, method_label, weights_path, test_dir, m, ci, complexity, eff,
                          y_true, y_pred, y_prob, class_names, test_paths, norm_info, img_size)
    if do_print:
        print(f"    {method_label:<22}: macroF1={m['macro_f1']:.4f}  acc={m['overall_accuracy']:.4f}  "
              f"AUROC={m['auc_roc_macro']:.4f}  AUC-PR={m['auc_pr_macro']:.4f}  "
              f"kappa={m['kappa']:.4f}  mcc={m['mcc']:.4f}  balAcc={m['balanced_accuracy']:.4f}")
    return m, ci


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── 类别名与数量 ──
    class_names = list(args.class_names) if args.class_names else auto_detect_class_names(args.data_train)
    num_classes = len(class_names)
    if args.num_classes is not None and args.num_classes != num_classes:
        raise ValueError(f"--num-classes={args.num_classes} 但检测到 {num_classes} 个类别: {class_names}")
    print(f"[data] classes={class_names}  num_classes={num_classes}")

    # ── BoW 参数 ──
    scales = [int(s) for s in args.bow_scales.split(',')]
    bow_dim = len(scales) * args.bow_K
    scale_str = '_'.join(map(str, scales))

    # ── 输出根目录 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_dir, f"ml_classify_dual_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    # ── Step 0: 提取 / 缓存 BoW ──
    print("=" * 60)
    print("Step 0: 提取 LBP-BoW 特征")
    print("=" * 60)
    train_hash = hashlib.md5(args.data_train.encode()).hexdigest()[:8]
    test_hash = hashlib.md5(args.data_test.encode()).hexdigest()[:8]
    train_cache = os.path.join(args.bow_cache_dir, f'train_K{args.bow_K}_s{scale_str}_{train_hash}.pkl')
    test_cache = os.path.join(args.bow_cache_dir, f'test_K{args.bow_K}_s{scale_str}_{test_hash}.pkl')
    extract_and_cache_bow(args.data_train, train_cache, scales=scales, K=args.bow_K)
    extract_and_cache_bow(args.data_test, test_cache, scales=scales, K=args.bow_K)
    bow_cache_train = BOWFeatureCache(train_cache, bow_dim=bow_dim)
    bow_cache_test = BOWFeatureCache(test_cache, bow_dim=bow_dim)
    print()

    # ── Step 1: 构造双流模型 (classifier → Identity 作特征提取器) ──
    print("=" * 60)
    print("Step 1: 加载 dual_stream 特征提取器")
    print("=" * 60)
    model = create_model(backbone_name=args.model, num_classes=num_classes, bow_dim=bow_dim).to(device)
    feat_dim = model.classifier[0].in_features   # = 2*d_model
    model.classifier = nn.Identity()
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    own = model.state_dict()
    ckpt = {k: v for k, v in ckpt.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    print(f"  权重: {args.weights}")
    print(f"  backbone={args.model}  特征维度={feat_dim}  bow_dim={bow_dim}  类别数={num_classes}")
    print(f"  输出目录: {base_dir}")
    print()

    # ── Step 2: 加载数据 + 归一化 (与训练 run_meta 同口径) ──
    print("Step 2: 加载数据集")
    train_paths, train_labels = read_split_train_data(args.data_train)
    test_paths, test_labels = read_split_val_data(args.data_test)
    norm_mean, norm_std, norm_src = resolve_norm(args.weights, args.norm_mean, args.norm_std)
    norm_info = (norm_mean, norm_std, norm_src)
    print(f"  norm: {norm_src}  mean={[round(x,4) for x in norm_mean]}  std={[round(x,4) for x in norm_std]}")
    transform = build_val_transform(args.img_size, norm_mean, norm_std)

    train_dataset = MyDataSetBOW(images_path=train_paths, images_class=train_labels,
                                 transform=transform, bow_cache=bow_cache_train)
    test_dataset = MyDataSetBOW(images_path=test_paths, images_class=test_labels,
                                transform=transform, bow_cache=bow_cache_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=test_dataset.collate_fn)
    print(f"  训练集: {len(train_paths)} 张   测试集: {len(test_paths)} 张   img_size={args.img_size}")
    print()

    # ── Step 3: 提取特征 ──
    print("Step 3: 提取特征")
    train_features, train_labels_np = extract_features(model, train_loader, device)
    t0 = time.time()
    test_features, test_labels_np = extract_features(model, test_loader, device)
    feat_extract_time = time.time() - t0
    feat_extract_per_img_ms = feat_extract_time / len(test_paths) * 1000
    print(f"  train_features={train_features.shape}  test_features={test_features.shape}")
    print(f"  测试集特征提取: {feat_extract_time:.3f}s ({feat_extract_per_img_ms:.4f} ms/张)")
    np.save(os.path.join(base_dir, "train_features.npy"), train_features)
    np.save(os.path.join(base_dir, "test_features.npy"), test_features)
    print(f"  [saved] train_features.npy, test_features.npy")
    print()

    # ── Step 4: 标准化特征 ──
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    # ── 复杂度 (双流模型，共享给所有方法作特征提取成本) ──
    complexity = compute_dual_complexity(model, args.img_size, bow_dim, device)
    print(f"[complexity] MACs={complexity['macs']} FLOPs={complexity['flops']} Params={complexity['params']}")
    print()

    # ── Step 5: 端到端 dual_stream 基准 ──
    print("Step 4: 端到端 dual_stream 基准")
    model_e2e = create_model(backbone_name=args.model, num_classes=num_classes, bow_dim=bow_dim).to(device)
    e2e_ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    if isinstance(e2e_ckpt, dict) and "model" in e2e_ckpt and isinstance(e2e_ckpt["model"], dict):
        e2e_ckpt = e2e_ckpt["model"]
    e2e_own = model_e2e.state_dict()
    model_e2e.load_state_dict(
        {k: v for k, v in e2e_ckpt.items() if k in e2e_own and e2e_own[k].shape == v.shape},
        strict=False)
    model_e2e.to(device).eval()

    e2e_probs, e2e_preds, t0 = [], [], time.time()
    with torch.no_grad():
        for images, bow_feats, labels in test_loader:
            logits = model_e2e(images.to(device), bow_feats.to(device))
            e2e_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            e2e_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    e2e_time = time.time() - t0
    e2e_probs = np.concatenate(e2e_probs, axis=0)
    e2e_preds = np.array(e2e_preds)
    e2e_eff = make_eff(e2e_time, e2e_time, len(test_paths))
    print(f"  端到端推理: {e2e_time:.3f}s ({e2e_time/len(test_paths)*1000:.4f} ms/张)")
    e2e_metrics, e2e_ci = run_method(
        "dual_stream-E2E", "dual_stream_E2E", test_labels_np, e2e_preds, e2e_probs,
        test_paths, base_dir, num_classes, class_names, complexity, e2e_eff,
        args.weights, args.data_test, args.img_size, norm_info, args, do_print=False)
    evaluate.print_report("dual_stream-E2E", len(test_paths), e2e_metrics, class_names,
                          complexity, e2e_eff, e2e_ci)
    print()

    # ── Step 6~: ML 分类器 ──
    def _ml_eff(predict_time):
        # pure = ML 预测耗时；e2e = 特征提取 + 预测
        return make_eff(predict_time, feat_extract_time + predict_time, len(test_paths))

    methods = []  # (label, dir_name, metrics)

    print("Step 5: SVM (RBF)")
    svm_rbf = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
    svm_rbf.fit(train_features_scaled, train_labels_np)
    t0 = time.time()
    preds = svm_rbf.predict(test_features_scaled); probs = svm_rbf.predict_proba(test_features_scaled)
    dt = time.time() - t0
    m, _ = run_method("SVM-RBF", "SVM_RBF", test_labels_np, preds, probs, test_paths,
                      base_dir, num_classes, class_names, complexity, _ml_eff(dt),
                      args.weights, args.data_test, args.img_size, norm_info, args)
    methods.append(("SVM-RBF", "SVM_RBF", m))

    print("Step 6: SVM (Linear)")
    svm_lin = SVC(kernel='linear', probability=True, C=1.0)
    svm_lin.fit(train_features_scaled, train_labels_np)
    t0 = time.time()
    preds = svm_lin.predict(test_features_scaled); probs = svm_lin.predict_proba(test_features_scaled)
    dt = time.time() - t0
    m, _ = run_method("SVM-Linear", "SVM_Linear", test_labels_np, preds, probs, test_paths,
                      base_dir, num_classes, class_names, complexity, _ml_eff(dt),
                      args.weights, args.data_test, args.img_size, norm_info, args)
    methods.append(("SVM-Linear", "SVM_Linear", m))

    print("Step 7: Logistic Regression")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(train_features_scaled, train_labels_np)
    t0 = time.time()
    preds = lr.predict(test_features_scaled); probs = lr.predict_proba(test_features_scaled)
    dt = time.time() - t0
    m, _ = run_method("LogisticRegression", "LogisticRegression", test_labels_np, preds, probs,
                      test_paths, base_dir, num_classes, class_names, complexity, _ml_eff(dt),
                      args.weights, args.data_test, args.img_size, norm_info, args)
    methods.append(("LogReg", "LogisticRegression", m))

    print("Step 8: KNN (k=5)")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_features_scaled, train_labels_np)
    t0 = time.time()
    preds = knn.predict(test_features_scaled); probs = knn.predict_proba(test_features_scaled)
    dt = time.time() - t0
    m, _ = run_method("KNN", "KNN", test_labels_np, preds, probs, test_paths,
                      base_dir, num_classes, class_names, complexity, _ml_eff(dt),
                      args.weights, args.data_test, args.img_size, norm_info, args)
    methods.append(("KNN", "KNN", m))

    print("Step 9: Random Forest")
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(train_features_scaled, train_labels_np)
    t0 = time.time()
    preds = rf.predict(test_features_scaled); probs = rf.predict_proba(test_features_scaled)
    dt = time.time() - t0
    m, _ = run_method("RandomForest", "RandomForest", test_labels_np, preds, probs, test_paths,
                      base_dir, num_classes, class_names, complexity, _ml_eff(dt),
                      args.weights, args.data_test, args.img_size, norm_info, args)
    methods.append(("RF", "RandomForest", m))

    print("Step 10: XGBoost")
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                        eval_metric='mlogloss', verbosity=0)
    xgb.fit(train_features_scaled, train_labels_np)
    t0 = time.time()
    preds = xgb.predict(test_features_scaled); probs = xgb.predict_proba(test_features_scaled)
    dt = time.time() - t0
    m, _ = run_method("XGBoost", "XGBoost", test_labels_np, preds, probs, test_paths,
                      base_dir, num_classes, class_names, complexity, _ml_eff(dt),
                      args.weights, args.data_test, args.img_size, norm_info, args)
    methods.append(("XGBoost", "XGBoost", m))

    print("Step 11: LightGBM")
    lgbm = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          random_state=42, verbose=-1)
    lgbm.fit(train_features_scaled, train_labels_np)
    t0 = time.time()
    preds = lgbm.predict(test_features_scaled); probs = lgbm.predict_proba(test_features_scaled)
    dt = time.time() - t0
    m, _ = run_method("LightGBM", "LightGBM", test_labels_np, preds, probs, test_paths,
                      base_dir, num_classes, class_names, complexity, _ml_eff(dt),
                      args.weights, args.data_test, args.img_size, norm_info, args)
    methods.append(("LightGBM", "LightGBM", m))

    print("Step 12: Voting Ensemble (LR + SVM_Linear + RF)")
    voting = VotingClassifier(
        estimators=[('lr', lr), ('svm_linear', svm_lin), ('rf', rf)], voting='soft')
    voting.fit(train_features_scaled, train_labels_np)
    t0 = time.time()
    preds = voting.predict(test_features_scaled); probs = voting.predict_proba(test_features_scaled)
    dt = time.time() - t0
    m, _ = run_method("VotingEnsemble", "VotingEnsemble", test_labels_np, preds, probs, test_paths,
                      base_dir, num_classes, class_names, complexity, _ml_eff(dt),
                      args.weights, args.data_test, args.img_size, norm_info, args)
    methods.append(("Voting", "VotingEnsemble", m))

    all_methods = [("dual_stream-E2E", "dual_stream_E2E", e2e_metrics)] + methods
    print()

    # ── 汇总表 + CSV ──
    col_w = 13
    metric_keys = [
        ("macro_f1", "Macro-F1"), ("overall_accuracy", "Accuracy"),
        ("auc_roc_macro", "AUROC(OvR)"), ("auc_pr_macro", "AUC-PR"),
        ("balanced_accuracy", "BalancedAcc"), ("kappa", "Kappa"), ("mcc", "MCC"),
        ("macro_precision", "Macro-Prec"), ("macro_recall", "Macro-Sens"),
        ("macro_specificity", "Macro-Spec"), ("weighted_f1", "Weighted-F1"),
    ]
    table_w = 20 + col_w * len(all_methods)
    print("=" * table_w)
    print(f"{'Results Comparison (metrics: classifyML/evaluate.py)':^{table_w}}")
    print("=" * table_w)
    print(f"{'Metric':<20}" + "".join(f"{m[0]:>{col_w}}" for m in all_methods))
    print("-" * table_w)
    for key, label in metric_keys:
        print(f"{label:<20}" + "".join(f"{m[2][key]:>{col_w}.4f}" for m in all_methods))
    print("=" * table_w)

    summary_path = os.path.join(base_dir, "comparison_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + [m[1] for m in all_methods])
        for key, label in metric_keys:
            w.writerow([label] + [f"{m[2][key]:.6f}" for m in all_methods])
    print(f"\n[saved] {summary_path}")
    print(f"\n所有结果已保存至: {base_dir}/")
    for _, dir_name, _ in all_methods:
        print(f"  ├── {dir_name:<22} (predictions.csv / results.json / evaluation_report.md)")
    print("  ├── train_features.npy / test_features.npy")
    print("  └── comparison_summary.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dual_stream 特征提取 + ML 多分类 (指标对齐 evaluate.py)')
    parser.add_argument('--weights', type=str, required=True, help='训练好的 dual_stream 权重路径')
    parser.add_argument('--model', type=str, default='fast_t0',
                        help='骨干模型名 (须与训练时 --model 一致)')
    parser.add_argument('--data-train', type=str, required=True, help='训练集目录')
    parser.add_argument('--data-test', type=str, required=True, help='测试集目录')
    parser.add_argument('--img_size', type=int, default=224,
                        help='图像输入尺寸，须与训练时 --img_size 一致')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=str, default='./bow_fusion/outputs')
    parser.add_argument('--class-names', nargs='+', type=str, default=None,
                        help='类别名称列表(空格分隔)；未提供则从训练集目录自动检测')
    parser.add_argument('--num-classes', type=int, default=None, help='类别数量(可选，用于校验)')
    parser.add_argument('--norm-mean', type=float, nargs=3, default=None, help='覆盖归一化均值')
    parser.add_argument('--norm-std', type=float, nargs=3, default=None, help='覆盖归一化标准差')
    parser.add_argument('--bootstrap', action='store_true', help='开启核心指标 95%% bootstrap CI')
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=42)
    # BoW 参数
    parser.add_argument('--bow-K', type=int, default=256, help='BoW codebook size (须与训练一致)')
    parser.add_argument('--bow-scales', type=str, default='16,32,64', help='多尺度(须与训练一致)')
    parser.add_argument('--bow-cache-dir', type=str, default='./bow_fusion/bow_cache')
    args = parser.parse_args()
    main(args)
