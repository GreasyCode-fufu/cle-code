"""
三分类模型评估脚本（规范版 v2）
==============================
按模型名切换架构，对测试集做全面评估。

主指标（不平衡医学数据首选 macro-F1）：
  - Macro-F1   : 各类 F1 等权平均（sklearn 标准，主指标）
  - Accuracy   : 总正确 / 总样本
  - Macro-AUROC: One-vs-Rest 宏平均

临床指标（三分类按 One-vs-Rest，每类各有一组）：
  - Sensitivity = Recall（该类 TP/(TP+FN)）
  - Specificity = 该类 TN/(TN+FP)
  另给 macro 平均的 Sensitivity / Specificity 作总体参考。

复杂度/效率：
  - MACs / FLOPs / Params（thop，仅与架构+输入尺寸有关，与数据集无关）
  - Pure-Inference FPS / End-to-End FPS（受 GPU 负载影响，为参考量级）

归一化：默认从权重旁 run_meta.json 自动读取，保证训练/评估一致；
        读不到再用 --norm-mean/--norm-std，最后兜底 ImageNet。

使用：
  python evaluate.py --model resnet34 \
      --weights weights_3class_hq_seed42/resnet34/resnet34_3class_hq_seed42_best.pth \
      --data-path /home/login/cle/sunfuxiang/datasets/3class/test
  # 论文最终模型加置信区间：  ... --bootstrap
"""

import os
import csv
import json
import time
import argparse
import warnings
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from sklearn.metrics import (
    confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")

# ========================== 模型注册表 ==========================
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
    "pvt_tiny":         ("models.Pvt", "pvt_tiny"),
    "vanillanet_5":     ("models.VanilaNet", "vanillanet_5"),
    "alexnet":          ("models.AlexNet", "alexnet"),
    "convnext_tiny":    ("models.Convnext", "convnext_tiny"),
    "convnext_small":   ("models.Convnext", "convnext_small"),
    "efficientnet_b0":  ("models.EfficientNet", "efficientnet_b0"),
    "mobilenet_v2":     ("models.MobileNetV2", "mobilenet_v2"),
    "swin_tiny_patch4_window7_224": ("models.Swin", "swin_tiny_patch4_window7_224"),
    "swin_tiny_vanilla_patch4_window7_224": ("models.Swin_vanilla", "swin_tiny_patch4_window7_224"),
    "vit_base_patch16_224":         ("models.Vit", "vit_base_patch16_224"),
    "resnet34_degconv_sa": ("models.optimize.resnet34_msinit", "resnet34_degconv_sa"),
    "resnet34_sa": ("models.optimize.resnet34_msinit", "resnet34_sa"),
    "resnet34_degconv_cpca": ("models.optimize.resnet34_CPCA", "resnet34_degconv_cpca"),
    "resnet34_degconv_cpca_v1": ("models.optimize.resnet34_CPCA_v1", "resnet34_degconv_cpca"),
    "resnet34_degconv_cpca_v2": ("models.optimize.resnet34_CPCA_v2", "resnet34_degconv_cpca"),
    "resnet34_degconv_cpca_v3": ("models.optimize.test", "resnet34_cpca"),
    "resnet34_se": ("models.optimize.resnet34_se", "resnet34_se"),
    "resnet50_se": ("models.optimize.resnet34_se", "resnet50_se"),
    "resnet34_s2d_v1": ("models.optimize.resnet34_s2d", "resnet34_s2d_v1"),
    "resnet34_s2d_v2": ("models.optimize.resnet34_s2d", "resnet34_s2d_v2"),
    "resnet34_msec_v1":      ("models.optimize.resnet34_msec", "resnet34_msec_v1"),
    "resnet34_msec_strips":  ("models.optimize.resnet34_msec", "resnet34_msec_strips"),
    "resnet34_msec_edge":    ("models.optimize.resnet34_msec", "resnet34_msec_edge"),
    "resnet34_msec_dil":     ("models.optimize.resnet34_msec", "resnet34_msec_dil"),
    "resnet34_msec_v1_p234": ("models.optimize.resnet34_msec", "resnet34_msec_v1_p234"),
    "resnet34_msec_dil_p234": ("models.optimize.resnet34_dilgate", "resnet34_msec_dil_p234"),
    "resnet34_msec_dil234":   ("models.optimize.resnet34_dilgate", "resnet34_msec_dil234"),
    "resnet34_pfesa":         ("models.optimize.resnet34_dilgate", "resnet34_pfesa"),
    "resnet34_pfesa_dil":     ("models.optimize.resnet34_dilgate", "resnet34_pfesa_dil"),
    "resnet34_dil234w": ("models.optimize.resnet34_dilgate", "resnet34_dil234w"),
    "resnet34_dil2346": ("models.optimize.resnet34_dilgate", "resnet34_dil2346"),
    "resnet34_dil123": ("models.optimize.resnet34_dilgate", "resnet34_dil123"),
    "resnet34_dil_r8": ("models.optimize.resnet34_dilgate", "resnet34_dil_r8"),
    "resnet34_dil234_l4": ("models.optimize.resnet34_dilgate", "resnet34_dil234_l4"),
    "resnet34_dil234_l3": ("models.optimize.resnet34_dilgate", "resnet34_dil234_l3"),
    "resnet34_dil234_k5": ("models.optimize.resnet34_dilgate", "resnet34_dil234_k5"),
    "resnet34_dil234_sg": ("models.optimize.resnet34_dilgate", "resnet34_dil234_sg"),
    "resnet34_dil23_r": ("models.optimize.resnet34_dilgate", "resnet34_dil23_r"),
    "resnet34_dil34_r": ("models.optimize.resnet34_dilgate", "resnet34_dil34_r"),
    "resnet34_dil234_prog": ("models.optimize.resnet34_dilgate", "resnet34_dil234_prog"),
    "resnet34_dil234_mix": ("models.optimize.resnet34_dilgate", "resnet34_dil234_mix"),
    "resnet34_dil234_id": ("models.optimize.resnet34_dilgate", "resnet34_dil234_id"),
    "resnet34_dil234_x2": ("models.optimize.resnet34_dilgate", "resnet34_dil234_x2"),
    "resnet34_atrous4": ("models.optimize.resnet34_atrous", "resnet34_atrous4"),
    "resnet34_atrous4_dil234": ("models.optimize.resnet34_atrous", "resnet34_atrous4_dil234"),
    "resnet34_lib_haar": ("models.optimize.lib_variants", "resnet34_lib_haar"),
    "resnet34_lib_msctx": ("models.optimize.lib_variants", "resnet34_lib_msctx"),
    "resnet34_lib_lcattn": ("models.optimize.lib_variants", "resnet34_lib_lcattn"),
    "resnet34_atrous34": ("models.optimize.resnet34_atrous", "resnet34_atrous34"),
}
# ===============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ───────────────────── 归一化解析（防错配）─────────────────────

def resolve_norm(args):
    if args.norm_mean is not None and args.norm_std is not None:
        return list(args.norm_mean), list(args.norm_std), "cli"
    meta_path = args.run_meta or os.path.join(
        os.path.dirname(os.path.abspath(args.weights)), "run_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "norm_mean" in meta and "norm_std" in meta:
            src = meta.get("norm_source", "run_meta")
            return meta["norm_mean"], meta["norm_std"], f"run_meta.json({src})"
    print("  [警告] 未找到 run_meta.json 也未传 --norm-mean/std，回退 ImageNet。"
          "若训练用数据集自算 mean/std，请显式传参！")
    return IMAGENET_MEAN, IMAGENET_STD, "imagenet_fallback"


# ───────────────────── 模型加载 + 复杂度 ─────────────────────

def load_model(weights_path, device, model_name, num_classes):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: '{model_name}'，可用: "
                         + ", ".join(sorted(MODEL_REGISTRY.keys())))
    module_name, func_name = MODEL_REGISTRY[model_name]
    create_model = getattr(__import__(module_name, fromlist=[func_name]), func_name)
    model = create_model(num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    valid_keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in valid_keys}
    missing = model.load_state_dict(state_dict, strict=False)
    if missing.missing_keys:
        print(f"  [提示] 加载缺失 {len(missing.missing_keys)} 键（首个: "
              f"{missing.missing_keys[0]}）—— swin 等自定义层属正常，其它请核查")
    model.to(device)
    model.eval()
    return model


def compute_complexity(model, img_size, device):
    """MACs / FLOPs / Params（thop）。失败则返回 N/A 不中断评估。"""
    try:
        from thop import profile, clever_format
        dummy = torch.zeros((1, 3, img_size, img_size)).to(device)
        macs_raw, params_raw = profile(model, inputs=(dummy,), verbose=False)
        flops_raw = 2 * macs_raw
        macs_str, params_str = clever_format([macs_raw, params_raw], "%.2f")
        flops_str, _ = clever_format([flops_raw, 0], "%.2f")
        # 清理 thop 注入的 buffer，避免污染后续 state_dict
        for mod in model.modules():
            for attr in ("total_ops", "total_params"):
                if hasattr(mod, attr):
                    delattr(mod, attr)
        return {"macs": macs_str, "flops": flops_str, "params": params_str,
                "params_raw": int(params_raw), "macs_raw": int(macs_raw)}
    except Exception as e:
        print(f"  [提示] 复杂度计算跳过（thop 不可用或不支持该模型）: {e}")
        return {"macs": "N/A", "flops": "N/A", "params": "N/A",
                "params_raw": 0, "macs_raw": 0}


# ───────────────────── 数据与推理（计时）─────────────────────

def load_test_data(test_dir, class_names):
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
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])


def run_inference(model, img_paths, transform, batch_size, device):
    """返回 (概率矩阵, 效率字典)。区分纯推理耗时与端到端耗时。"""
    use_cuda = (device.type == "cuda")
    all_probs = []
    pure_infer_time = 0.0
    t_e2e = time.time()
    for i in range(0, len(img_paths), batch_size):
        batch_imgs = [transform(Image.open(p).convert("RGB"))
                      for p in img_paths[i:i + batch_size]]
        batch_tensor = torch.stack(batch_imgs).to(device)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        if use_cuda:
            torch.cuda.synchronize()
        pure_infer_time += time.time() - t0
        all_probs.append(probs)
    e2e_time = time.time() - t_e2e
    n = len(img_paths)
    eff = {
        "pure_infer_time": pure_infer_time,
        "pure_fps": n / pure_infer_time if pure_infer_time > 0 else 0.0,
        "pure_ms_per_img": pure_infer_time / n * 1000 if n else 0.0,
        "e2e_time": e2e_time,
        "e2e_fps": n / e2e_time if e2e_time > 0 else 0.0,
        "total_samples": n,
    }
    return np.concatenate(all_probs, axis=0), eff


# ───────────────────── 指标计算 ─────────────────────

def compute_metrics(y_true, y_pred, y_prob, num_classes, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    overall_acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = float(np.mean(recall_score(y_true, y_pred, average=None, zero_division=0)))

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class = {}
    spec_list = []
    for c in range(num_classes):
        yt = (y_true == c).astype(int)
        yp = (y_pred == c).astype(int)
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0       # Sensitivity
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0       # Specificity
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        spec_list.append(spec)
        per_class[class_names[c]] = {
            "precision": prec, "recall": rec, "sensitivity": rec,
            "specificity": spec, "f1": f1, "npv": npv,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "support": int(yt.sum()),
        }

    macro_sensitivity = macro_recall                 # = 各类 recall 均值
    macro_specificity = float(np.mean(spec_list))    # 各类 specificity 均值

    auc_roc_per_class, auc_pr_per_class = {}, {}
    y_true_bin = np.eye(num_classes)[y_true]
    try:
        auc_ovr = roc_auc_score(y_true, y_prob, multi_class="ovr", average=None)
        for c in range(num_classes):
            auc_roc_per_class[class_names[c]] = float(auc_ovr[c])
    except ValueError:
        pass
    for c in range(num_classes):
        try:
            auc_pr_per_class[class_names[c]] = float(
                average_precision_score(y_true_bin[:, c], y_prob[:, c]))
        except ValueError:
            pass
    auc_roc_macro = (roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                     if len(np.unique(y_true)) == num_classes else float("nan"))
    auc_pr_macro = (float(np.mean(list(auc_pr_per_class.values())))
                    if auc_pr_per_class else float("nan"))

    return {
        "macro_f1": macro_f1, "overall_accuracy": overall_acc,
        "auc_roc_macro": auc_roc_macro, "auc_pr_macro": auc_pr_macro,
        "weighted_f1": weighted_f1, "balanced_accuracy": balanced_acc,
        "kappa": kappa, "mcc": mcc,
        "macro_precision": macro_precision, "macro_recall": macro_recall,
        "macro_sensitivity": macro_sensitivity, "macro_specificity": macro_specificity,
        "weighted_precision": weighted_precision, "weighted_recall": weighted_recall,
        "per_class": per_class, "confusion_matrix": cm,
        "auc_roc_per_class": auc_roc_per_class, "auc_pr_per_class": auc_pr_per_class,
    }


def bootstrap_ci(y_true, y_prob, num_classes, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    macro_f1_l, acc_l, auc_l, sens_l, spec_l = [], [], [], [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        yt, pp = y_true[idx], y_prob[idx]
        yp = np.argmax(pp, axis=1)
        if len(np.unique(yt)) < num_classes:
            continue
        macro_f1_l.append(f1_score(yt, yp, average="macro", zero_division=0))
        acc_l.append(accuracy_score(yt, yp))
        sens_l.append(recall_score(yt, yp, average="macro", zero_division=0))
        specs = []
        for c in range(num_classes):
            ytc = (yt == c).astype(int); ypc = (yp == c).astype(int)
            tn = int(np.sum((ytc == 0) & (ypc == 0)))
            fp = int(np.sum((ytc == 0) & (ypc == 1)))
            specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        spec_l.append(np.mean(specs))
        try:
            auc_l.append(roc_auc_score(yt, pp, multi_class="ovr", average="macro"))
        except ValueError:
            pass

    def _ci(lst):
        return (float(np.percentile(lst, 2.5)), float(np.percentile(lst, 97.5))) \
            if len(lst) >= 10 else (float("nan"), float("nan"))
    return {"macro_f1": _ci(macro_f1_l), "accuracy": _ci(acc_l),
            "auc_roc_macro": _ci(auc_l), "macro_sensitivity": _ci(sens_l),
            "macro_specificity": _ci(spec_l)}


# ───────────────────── 规范化终端输出 ─────────────────────

def print_report(model_name, n_total, m, class_names, complexity, eff, ci=None):
    W = 64
    print("\n" + "=" * W)
    print(f"  评估结果 — {model_name}  (test={n_total} 张)")
    print("=" * W)

    print("  【主指标】")
    if ci:
        print(f"  Macro-F1         : {m['macro_f1']:.4f}  "
              f"[{ci['macro_f1'][0]:.4f}, {ci['macro_f1'][1]:.4f}]   ← 主指标")
        print(f"  Accuracy         : {m['overall_accuracy']:.4f}  "
              f"[{ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f}]")
        print(f"  Macro-AUROC(OvR) : {m['auc_roc_macro']:.4f}  "
              f"[{ci['auc_roc_macro'][0]:.4f}, {ci['auc_roc_macro'][1]:.4f}]")
    else:
        print(f"  Macro-F1         : {m['macro_f1']:.4f}   ← 主指标(类别不平衡首选)")
        print(f"  Accuracy         : {m['overall_accuracy']:.4f}")
        print(f"  Macro-AUROC(OvR) : {m['auc_roc_macro']:.4f}")

    print("-" * W)
    print("  【补充总体指标】")
    print(f"  Weighted-F1      : {m['weighted_f1']:.4f}")
    print(f"  Balanced Acc     : {m['balanced_accuracy']:.4f}")
    print(f"  Macro-Sensitivity: {m['macro_sensitivity']:.4f}   (= 各类 Recall 均值)")
    print(f"  Macro-Specificity: {m['macro_specificity']:.4f}")
    print(f"  Macro-Precision  : {m['macro_precision']:.4f}")
    print(f"  AUC-PR (Macro)   : {m['auc_pr_macro']:.4f}")
    print(f"  Cohen's Kappa    : {m['kappa']:.4f}")
    print(f"  MCC              : {m['mcc']:.4f}")

    print("-" * W)
    print("  【每类指标】 Sensitivity / Specificity / Precision / F1 / AUROC")
    for c in class_names:
        pc = m["per_class"][c]
        auc = m["auc_roc_per_class"].get(c, float("nan"))
        print(f"  {c:<22}: {pc['sensitivity']:.3f} / {pc['specificity']:.3f} / "
              f"{pc['precision']:.3f} / {pc['f1']:.3f} / {auc:.3f}  (n={pc['support']})")

    print("-" * W)
    print("  【模型复杂度】(仅与架构+输入尺寸有关，与数据集无关)")
    print(f"  MACs : {complexity['macs']}    FLOPs : {complexity['flops']}    "
          f"Params : {complexity['params']}")

    print("  【推理效率】(受 GPU 负载影响，为参考量级)")
    print(f"  Pure FPS : {eff['pure_fps']:.1f}  ({eff['pure_ms_per_img']:.2f} ms/张)    "
          f"End-to-End FPS : {eff['e2e_fps']:.1f}")

    print("-" * W)
    print("  【混淆矩阵】(行=真实, 列=预测)")
    cm = m["confusion_matrix"]
    short = [c[:8] for c in class_names]
    print("  " + " " * 14 + "".join(f"{s:>10}" for s in short))
    for i, c in enumerate(class_names):
        row = "".join(f"{cm[i, j]:>10}" for j in range(len(class_names)))
        print(f"  {c[:12]:<14}{row}")
    print("=" * W)


# ───────────────────── 报告文件 ─────────────────────

def save_outputs(save_dir, model_name, weights_path, test_dir, m, ci, complexity, eff,
                 y_true, y_pred, y_prob, class_names, img_paths, norm_info, img_size):
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "true_label", "pred_label"] +
                   [f"prob_{c}" for c in class_names])
        for i in range(len(y_true)):
            w.writerow([img_paths[i], class_names[y_true[i]], class_names[y_pred[i]]] +
                       [f"{p:.6f}" for p in y_prob[i]])

    out = {
        "model": model_name, "weights": os.path.abspath(weights_path),
        "test_dir": test_dir, "img_size": img_size,
        "norm_mean": norm_info[0], "norm_std": norm_info[1], "norm_source": norm_info[2],
        "n_total": len(y_true), "class_names": class_names,
        "macro_f1": m["macro_f1"], "accuracy": m["overall_accuracy"],
        "auc_roc_macro": m["auc_roc_macro"], "weighted_f1": m["weighted_f1"],
        "balanced_accuracy": m["balanced_accuracy"],
        "macro_sensitivity": m["macro_sensitivity"],
        "macro_specificity": m["macro_specificity"],
        "kappa": m["kappa"], "mcc": m["mcc"],
        "complexity": complexity, "efficiency": eff,
        "per_class": {c: m["per_class"][c] for c in class_names},
        "confusion_matrix": m["confusion_matrix"].tolist(),
    }
    if ci:
        out["bootstrap_95ci"] = ci
    with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    L = [f"# {model_name} 评估报告", "",
         f"- 权重: `{weights_path}`",
         f"- 测试集: `{test_dir}`  (共 {len(y_true)} 张)",
         f"- 归一化: {norm_info[2]}  mean={[round(x,4) for x in norm_info[0]]}",
         f"- img_size: {img_size}", "",
         "## 主指标", "", "| 指标 | 值 |", "|---|---|",
         f"| **Macro-F1** | {m['macro_f1']:.4f} |",
         f"| Accuracy | {m['overall_accuracy']:.4f} |",
         f"| Macro-AUROC (OvR) | {m['auc_roc_macro']:.4f} |", "",
         "## 补充总体指标", "", "| 指标 | 值 |", "|---|---|",
         f"| Weighted-F1 | {m['weighted_f1']:.4f} |",
         f"| Balanced Acc | {m['balanced_accuracy']:.4f} |",
         f"| Macro-Sensitivity | {m['macro_sensitivity']:.4f} |",
         f"| Macro-Specificity | {m['macro_specificity']:.4f} |",
         f"| Macro-Precision | {m['macro_precision']:.4f} |",
         f"| AUC-PR (Macro) | {m['auc_pr_macro']:.4f} |",
         f"| Cohen's Kappa | {m['kappa']:.4f} |",
         f"| MCC | {m['mcc']:.4f} |", "",
         "## 模型复杂度与推理效率", "", "| 指标 | 值 |", "|---|---|",
         f"| MACs | {complexity['macs']} |",
         f"| FLOPs | {complexity['flops']} |",
         f"| Params | {complexity['params']} |",
         f"| Pure-Inference FPS | {eff['pure_fps']:.1f} ({eff['pure_ms_per_img']:.2f} ms/张) |",
         f"| End-to-End FPS | {eff['e2e_fps']:.1f} |",
         f"| 测试样本数 | {eff['total_samples']} |", "",
         "## 每类指标 (One-vs-Rest)", "",
         "| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in class_names:
        pc = m["per_class"][c]
        auc = m["auc_roc_per_class"].get(c, float("nan"))
        ap = m["auc_pr_per_class"].get(c, float("nan"))
        L.append(f"| {c} | {pc['sensitivity']:.4f} | {pc['specificity']:.4f} | "
                 f"{pc['precision']:.4f} | {pc['f1']:.4f} | {pc['npv']:.4f} | "
                 f"{auc:.4f} | {ap:.4f} | {pc['support']} |")
    L += ["", "### 每类 TP/FP/TN/FN", ""]
    for c in class_names:
        pc = m["per_class"][c]
        L.append(f"- **{c}**: TP={pc['tp']} FP={pc['fp']} TN={pc['tn']} FN={pc['fn']}")
    L += ["", "## 混淆矩阵 (行=真实, 列=预测)", "",
          "| | " + " | ".join(class_names) + " |",
          "|---" * (len(class_names) + 1) + "|"]
    cm = m["confusion_matrix"]
    for i, c in enumerate(class_names):
        L.append(f"| {c} | " + " | ".join(str(cm[i, j]) for j in range(len(class_names))) + " |")
    if ci:
        L += ["", "## 核心指标 95% Bootstrap CI", "", "| 指标 | 95% CI |", "|---|---|"]
        for k, v in ci.items():
            L.append(f"| {k} | [{v[0]:.4f}, {v[1]:.4f}] |")
    L += ["", "## Classification Report (sklearn)", "", "```",
          classification_report(y_true, y_pred, target_names=class_names, digits=4), "```"]
    with open(os.path.join(save_dir, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ───────────────────── 主函数 ─────────────────────

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    class_names = args.class_names
    num_classes = len(class_names)

    norm_mean, norm_std, norm_src = resolve_norm(args)
    print(f"[norm] source={norm_src}  mean={[round(x,4) for x in norm_mean]} "
          f"std={[round(x,4) for x in norm_std]}  img_size={args.img_size}")

    print(f"[model] {args.model}  weights={args.weights}")
    model = load_model(args.weights, device, args.model, num_classes)

    complexity = compute_complexity(model, args.img_size, device)
    print(f"[complexity] MACs={complexity['macs']} FLOPs={complexity['flops']} "
          f"Params={complexity['params']}")

    img_paths, y_true = load_test_data(args.data_path, class_names)
    print(f"[data] test={len(img_paths)} 张  " +
          "  ".join(f"{c}={int(np.sum(y_true==i))}" for i, c in enumerate(class_names)))

    transform = build_transform(args.img_size, norm_mean, norm_std)
    y_prob, eff = run_inference(model, img_paths, transform, args.batch_size, device)
    y_pred = np.argmax(y_prob, axis=1)

    m = compute_metrics(y_true, y_pred, y_prob, num_classes, class_names)

    ci = None
    if args.bootstrap:
        print(f"[bootstrap] {args.n_bootstrap} 次重采样计算 95% CI ...")
        ci = bootstrap_ci(y_true, y_prob, num_classes,
                          n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed)

    print_report(args.model, len(y_true), m, class_names, complexity, eff, ci)

    save_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.weights)),
        f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_outputs(save_dir, args.model, args.weights, args.data_path, m, ci, complexity, eff,
                 y_true, y_pred, y_prob, class_names, img_paths,
                 (norm_mean, norm_std, norm_src), args.img_size)
    print(f"\n[saved] {save_dir}/  (results.json / predictions.csv / evaluation_report.md)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="3分类模型评估（规范版 v2）")
    p.add_argument("--model", type=str, required=True,
                   choices=sorted(MODEL_REGISTRY.keys()), help="模型名")
    p.add_argument("--weights", type=str, required=True, help="best 权重路径")
    p.add_argument("--data-path", type=str, required=True, help="测试集目录")
    p.add_argument("--class-names", type=str, nargs="+",
                   default=["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"],
                   help="类名(按 sorted 顺序，默认即 3class 实际类名)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--norm-mean", type=float, nargs=3, default=None,
                   help="覆盖归一化均值；默认读权重旁 run_meta.json")
    p.add_argument("--norm-std", type=float, nargs=3, default=None)
    p.add_argument("--run-meta", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--bootstrap", action="store_true",
                   help="开启核心指标 95%% bootstrap CI(较慢)")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    main(p.parse_args())
