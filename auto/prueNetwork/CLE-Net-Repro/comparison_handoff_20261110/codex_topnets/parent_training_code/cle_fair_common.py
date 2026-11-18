"""
Shared helpers for the CLE fair-comparison baselines (classifyML).
Single source of truth for: model registry + builder, val-set Youden's J threshold
search, and the full binary metric set — used by BOTH train_cle_fair.py and
evaluate_cle_fair.py so the --model names and the evaluation protocol stay aligned.
Original framework files (train.py, utils.py, evaluate_2class.py) are NOT modified.
"""
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, confusion_matrix, accuracy_score,
    cohen_kappa_score, precision_score, recall_score,
)

# ===================== Model registry =====================
# name (used by --model in BOTH train & eval) -> (module path, factory fn)
# Keys reuse evaluate_2class.py where they overlap (fast_t0, resnet34/50/101);
# densenet* is ADDED here (it exists in models/DenseNet.py but the original
# evaluate_2class.py registry did not register it).
MODEL_REGISTRY = {
    "fast_t0":     ("models.FasterNet", "fast_t0"),
    "resnet34":    ("models.Resnet",    "resnet34"),
    "resnet50":    ("models.Resnet",    "resnet50"),
    "resnet101":   ("models.Resnet",    "resnet101"),
    "resnet34_sa":   ("models.optimize.resnet34_msinit", "resnet34_sa"),
    "resnet34_degconv_sa": ("models.optimize.resnet34_msinit", "resnet34_degconv_sa"),
    "resnet34_degconv_cpca": ("models.optimize.resnet34_CPCA", "resnet34_degconv_cpca"),
    "resnet34_degconv_cpca_v1": ("models.optimize.resnet34_CPCA_v1", "resnet34_degconv_cpca"),
    "resnet34_degconv_cpca_v2": ("models.optimize.resnet34_CPCA_v2", "resnet34_degconv_cpca"),
    "resnet34_degconv_cpca_v3": ("models.optimize.test", "resnet34_cpca"),
    "densenet121": ("models.DenseNet",  "densenet121"),
    "densenet169": ("models.DenseNet",  "densenet169"),
    "densenet201": ("models.DenseNet",  "densenet201"),
    "densenet161": ("models.DenseNet",  "densenet161"),
    # ── Vision Transformer (models/Vit.py) ──
    "vit_base_patch16_224":          ("models.Vit", "vit_base_patch16_224"),
    "vit_base_patch16_224_in21k":    ("models.Vit", "vit_base_patch16_224_in21k"),
    "vit_base_patch32_224":          ("models.Vit", "vit_base_patch32_224"),
    "vit_base_patch32_224_in21k":    ("models.Vit", "vit_base_patch32_224_in21k"),
    "vit_large_patch16_224":         ("models.Vit", "vit_large_patch16_224"),
    "vit_large_patch16_224_in21k":   ("models.Vit", "vit_large_patch16_224_in21k"),
    "vit_large_patch32_224_in21k":   ("models.Vit", "vit_large_patch32_224_in21k"),
    "vit_huge_patch14_224_in21k":    ("models.Vit", "vit_huge_patch14_224_in21k"),
    # ── Swin Transformer (models/Swin.py) ──
    "swin_tiny_patch4_window7_224":       ("models.Swin", "swin_tiny_patch4_window7_224"),
    # Vanilla Swin-T for controlled validation. The legacy key above intentionally
    # remains mapped to the historical EFG-augmented implementation.
    "swin_tiny_vanilla_patch4_window7_224": ("models.Swin_vanilla", "swin_tiny_patch4_window7_224"),
    "swin_small_patch4_window7_224":      ("models.Swin", "swin_small_patch4_window7_224"),
    "swin_base_patch4_window7_224":       ("models.Swin", "swin_base_patch4_window7_224"),
    "swin_base_patch4_window12_384":      ("models.Swin", "swin_base_patch4_window12_384"),
    "swin_base_patch4_window7_224_in22k": ("models.Swin", "swin_base_patch4_window7_224_in22k"),
    "swin_base_patch4_window12_384_in22k":("models.Swin", "swin_base_patch4_window12_384_in22k"),
    "swin_large_patch4_window7_224_in22k":("models.Swin", "swin_large_patch4_window7_224_in22k"),
    "swin_large_patch4_window12_384_in22k":("models.Swin","swin_large_patch4_window12_384_in22k"),
    # ── Pyramid Vision Transformer (models/Pvt.py) ──
    "pvt_tiny":    ("models.Pvt", "pvt_tiny"),
    "pvt_small":   ("models.Pvt", "pvt_small"),
    "pvt_medium":  ("models.Pvt", "pvt_medium"),
    "pvt_large":   ("models.Pvt", "pvt_large"),
    "pvt_huge_v2": ("models.Pvt", "pvt_huge_v2"),
    # ── RepViT (models/RepViT.py) ──
    "repvit_m1":   ("models.RepViT", "repvit_m1"),
    "repvit_m2":   ("models.RepViT", "repvit_m2"),
    "repvit_m3":   ("models.RepViT", "repvit_m3"),
    # ── SE 注意力变体 (models/optimize/resnet34_se.py, 2026-10-23) ──
    "resnet34_se": ("models.optimize.resnet34_se", "resnet34_se"),
    "resnet50_se": ("models.optimize.resnet34_se", "resnet50_se"),
    # ── space_to_depth 变体: v1后置/v2前置, 版本化注册名, 权重目录自动分离 ──
    "resnet34_s2d_v1": ("models.optimize.resnet34_s2d", "resnet34_s2d_v1"),
    "resnet34_s2d_v2": ("models.optimize.resnet34_s2d", "resnet34_s2d_v2"),
    # ── MSEC 多尺度边缘-上下文门控残差 (models/optimize/resnet34_msec.py, 2026-10-23) ──
    "resnet34_msec_v1":      ("models.optimize.resnet34_msec", "resnet34_msec_v1"),
    "resnet34_msec_strips":  ("models.optimize.resnet34_msec", "resnet34_msec_strips"),
    "resnet34_msec_edge":    ("models.optimize.resnet34_msec", "resnet34_msec_edge"),
    "resnet34_msec_dil":     ("models.optimize.resnet34_msec", "resnet34_msec_dil"),
    "resnet34_msec_v1_p234": ("models.optimize.resnet34_msec", "resnet34_msec_v1_p234"),
    # ── Batch C: DilGate深挖 + PFESA (models/optimize/resnet34_dilgate.py) ──
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
    "resnet34_gmp": ("models.optimize.resnet34_redesign", "resnet34_gmp"),
    "resnet34_dshortcut": ("models.optimize.resnet34_redesign", "resnet34_dshortcut"),
    "resnet34_lib_msctx_haar": ("models.optimize.lib_variants", "resnet34_lib_msctx_haar"),
    "resnet34_lib_msctx_l4": ("models.optimize.lib_variants", "resnet34_lib_msctx_l4"),
    "resnet34_lib_msctx_l3": ("models.optimize.lib_variants", "resnet34_lib_msctx_l3"),
    "resnet34_hybrid_dil_msctx": ("models.optimize.lib_variants", "resnet34_hybrid_dil_msctx"),
    "resnet34_atrous34": ("models.optimize.resnet34_atrous", "resnet34_atrous34"),
    # ── ConvNeXt (models/Convnext.py, 本会话 baseline 方向, 2026-10-25) ──
    "convnext_tiny":  ("models.Convnext", "convnext_tiny"),
    "convnext_small": ("models.Convnext", "convnext_small"),
    # ── ConvNeXt-CLE 创新变体 (models/optimize/convnext_cle.py, 2026-10-25) ──
    # A=WDS小波细节stem B=WTB小波增强块(stage3/4) C=MSF多尺度融合头, 递进叠加
    "convnext_wds":           ("models.optimize.convnext_cle", "convnext_wds"),
    "convnext_wtb":           ("models.optimize.convnext_cle", "convnext_wtb"),
    "convnext_msf":           ("models.optimize.convnext_cle", "convnext_msf"),
    "convnext_wds_wtb":       ("models.optimize.convnext_cle", "convnext_wds_wtb"),
    "convnext_wds_wtb_msf":   ("models.optimize.convnext_cle", "convnext_wds_wtb_msf"),
    "convnext_wds_wtb_l1":    ("models.optimize.convnext_cle", "convnext_wds_wtb_l1"),
    "convnext_wds_wtb_s234":  ("models.optimize.convnext_cle", "convnext_wds_wtb_s234"),
    "convnext_wds_wtb_k3":    ("models.optimize.convnext_cle", "convnext_wds_wtb_k3"),
    "convnext_wtb_k3":        ("models.optimize.convnext_cle", "convnext_wtb_k3"),
    "convnext_wds_wtb_k3_msf": ("models.optimize.convnext_cle", "convnext_wds_wtb_k3_msf"),
    "convnext_wdd":           ("models.optimize.convnext_cle", "convnext_wdd"),
    "convnext_wds_msf":       ("models.optimize.convnext_cle", "convnext_wds_msf"),
    "convnext_wtb_msf":       ("models.optimize.convnext_cle", "convnext_wtb_msf"),
    # ── CLE-Net 第二代创新 (models/optimize/convnext_cle2.py, 2026-10-25) ──
    "convnext_egb":          ("models.optimize.convnext_cle2", "convnext_egb"),
    "convnext_scg":          ("models.optimize.convnext_cle2", "convnext_scg"),
    "convnext_hsc":          ("models.optimize.convnext_cle2", "convnext_hsc"),
    "convnext_egb_scg":      ("models.optimize.convnext_cle2", "convnext_egb_scg"),
    "convnext_egb_scg_hsd":  ("models.optimize.convnext_cle2", "convnext_egb_scg_hsd"),
    "convnext_egb_scg_hsdg": ("models.optimize.convnext_cle2", "convnext_egb_scg_hsdg"),
    "convnext_egb_scg_fbe": ("models.optimize.convnext_cle2", "convnext_egb_scg_fbe"),
    "convnext_egb_scg_fbe_all": ("models.optimize.convnext_cle2", "convnext_egb_scg_fbe_all"),
    # ── 频域辅助正则化 SR (训练期损失分支, 推理路径不变, 见 experiments/SR_DESIGN.md) ──
    "convnext_egb_scg_sr":   ("models.optimize.convnext_cle2", "convnext_egb_scg_sr"),
    "convnext_sr":           ("models.optimize.convnext_cle2", "convnext_sr"),
    "convnext_egb_sr":       ("models.optimize.convnext_cle2", "convnext_egb_sr"),
    "convnext_scg_sr":       ("models.optimize.convnext_cle2", "convnext_scg_sr"),
    # ── 主干容量缩放: S/B 底座的 CLE-Net + base 基线 ──
    "convnext_base":         ("models.Convnext", "convnext_base"),
    "convnext_small_egb_scg": ("models.optimize.convnext_cle2", "convnext_small_egb_scg"),
    "convnext_tiny_cle_dose": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_dose"),
    "convnext_tiny_cle_dose_late": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_dose_late"),
    "convnext_tiny_cle_ptg": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_ptg"),
    "convnext_tiny_cle_gds": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_gds"),
    "convnext_tiny_cle_dose_ptg": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_dose_ptg"),
    "convnext_tiny_cle_dose_late_ptg": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_dose_late_ptg"),
    "convnext_tiny_cle_dose_ptg_gds": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_dose_ptg_gds"),
    "convnext_tiny_cle_dose_late_ptg_gds": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_dose_late_ptg_gds"),
    "convnext_tiny_cle_ptg_gds": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_ptg_gds"),
    "convnext_tiny_cle_oca": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_oca"),
    "convnext_tiny_cle_oca_gds": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_oca_gds"),
    "convnext_tiny_cle_oca_ptg_gds": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_oca_ptg_gds"),
    "convnext_tiny_cle_dose_late_gds": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_dose_late_gds"),
    "convnext_tiny_cle_oca_ptg": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_oca_ptg"),
    "convnext_tiny_cle_oca_wide": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_oca_wide"),
    "convnext_tiny_cle_oca_block": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_oca_block"),
    "convnext_tiny_cle_oca_strip": ("models.optimize.convnext_cle_dsg", "convnext_tiny_cle_oca_strip"),
    "convnext_v2bc": ("models.optimize.convnext_sf_v2", "convnext_v2bc"),
    "cle_convnext_sn": ("models.optimize.cle_convnext_dpf", "cle_convnext_sn"),
    "convnext_base_egb_scg": ("models.optimize.convnext_cle2", "convnext_base_egb_scg"),
    "convnext_scg_hsc":      ("models.optimize.convnext_cle2", "convnext_scg_hsc"),
    # ── CLE-Net v2 结构级创新 (models/optimize/convnext_cle3.py, 2026-10-25) ──
    "convnext_os":           ("models.optimize.convnext_cle3", "convnext_os"),
    "convnext_dsh":          ("models.optimize.convnext_cle3", "convnext_dsh"),
    "convnext_os_scg":       ("models.optimize.convnext_cle3", "convnext_os_scg"),
    "convnext_os_scg_dsh":   ("models.optimize.convnext_cle3", "convnext_os_scg_dsh"),
    # ── CLE-Net v3 (models/optimize/convnext_cle4.py, 2026-10-25) ──
    "convnext_pld":          ("models.optimize.convnext_cle4", "convnext_pld"),
    "convnext_scg_pld":      ("models.optimize.convnext_cle4", "convnext_scg_pld"),
    # ── S-Net v4 (models/optimize/convnext_cle5.py, 2026-10-26) ──
    "convnext_esa":          ("models.optimize.convnext_cle5", "convnext_esa"),
    "convnext_esa_scg_hsc":  ("models.optimize.convnext_cle5", "convnext_esa_scg_hsc"),
    "convnext_egb_scg_hsc":  ("models.optimize.convnext_cle2", "convnext_egb_scg_hsc"),
    "convnext_egb_scg_msf":  ("models.optimize.convnext_cle2", "convnext_egb_scg_msf"),
    "convnext_wds_wdd_msf":   ("models.optimize.convnext_cle", "convnext_wds_wdd_msf"),
    # ── 外部创新网络 CLE TopoRel (models/toporel/, 源自 convnext_tiny_cle_toporel_relaux 项目
    #    逐字节复制, 仅改 import 路径; 两阶段: A=convnext_tiny_cle_toporel, B=relaux 冻结A训B) ──
    "convnext_tiny_cle_toporel":        ("models.toporel.topology", "convnext_tiny_cle_toporel"),
    "convnext_tiny_cle_toporel_relaux": ("models.toporel.topology", "convnext_tiny_cle_toporel_relaux"),
}

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def build_model(name, num_classes):
    """Dynamically import and instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        avail = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"unknown --model '{name}'. Available: {avail}")
    module_name, func_name = MODEL_REGISTRY[name]
    factory = getattr(__import__(module_name, fromlist=[func_name]), func_name)
    return factory(num_classes=num_classes)


# ===================== Data / transform =====================

def read_images_dir(root, class_names=None):
    """Scan root/<class>/*.{img}. class order = sorted subdir names (negative=0, positive=1).
    Returns (img_paths list, labels np.array, class_names list)."""
    assert os.path.isdir(root), f"dir not found: {root}"
    if class_names is None:
        class_names = sorted([d for d in os.listdir(root)
                              if os.path.isdir(os.path.join(root, d))])
    paths, labels = [], []
    for idx, cname in enumerate(class_names):
        cdir = os.path.join(root, cname)
        if not os.path.isdir(cdir):
            raise FileNotFoundError(f"class folder not found: {cdir}")
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(SUPPORTED_EXTS):
                paths.append(os.path.join(cdir, fn))
                labels.append(idx)
    return paths, np.array(labels), list(class_names)


def build_val_transform(img_size, norm_mean, norm_std):
    """Deterministic val/test transform: Resize((256,256)) -> CenterCrop(img_size) -> Norm.
    Forced-square resize matches CUFIT and train_cle_fair's val selection; shared by both
    training val-selection and test evaluation so the preprocessing never diverges."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])


@torch.no_grad()
def run_inference(model, img_paths, transform, batch_size, device):
    """Return softmax probability matrix of shape (N, num_classes)."""
    model.eval()
    use_cuda = (device.type == "cuda")
    all_probs = []
    for i in range(0, len(img_paths), batch_size):
        batch = [transform(Image.open(p).convert("RGB")) for p in img_paths[i:i + batch_size]]
        x = torch.stack(batch).to(device)
        with torch.amp.autocast('cuda', enabled=use_cuda):
            logits = model(x)
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# ===================== Threshold + metrics =====================

def youden_threshold(y_true, prob_pos, lo=0.05, hi=0.95, step=0.01):
    """Sweep thresholds on the positive-class probability; pick the one maximizing
    Youden's J = sensitivity + specificity - 1. Returns (best_thr, best_J)."""
    y_true = np.asarray(y_true)
    best_thr, best_j = 0.5, -1.0
    for thr in np.arange(lo, hi + 1e-9, step):
        pred = (prob_pos >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_thr = j, float(thr)
    return best_thr, best_j


def compute_binary_metrics(y_true, y_pred, y_prob):
    """Full binary metric set (aligned with CUFIT eval protocol).
    AUROC/AUC-PR depend only on y_prob (threshold-free);
    the rest depend on y_pred (i.e. on the chosen threshold). positive class = index 1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall of positive
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # recall of negative
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    paper_f1 = 2 * sensitivity * specificity / (sensitivity + specificity + 1e-8)
    try:
        auroc = roc_auc_score(y_true, y_prob[:, 1])
        auc_pr = average_precision_score(y_true, y_prob[:, 1])
    except Exception:
        auroc = auc_pr = float('nan')
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "precision_macro":   float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall_macro":      float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_macro":          float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "auroc":             float(auroc),
        "sensitivity":       float(sensitivity),
        "specificity":       float(specificity),
        "auroc_weighted":    float(auroc),   # binary: identical to auroc
        "auc_pr":            float(auc_pr),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa":             float(cohen_kappa_score(y_true, y_pred)),
        "mcc":               float(matthews_corrcoef(y_true, y_pred)),
        "precision_w":       float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall_w":          float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1_weighted":       float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        "paper_f1":          float(paper_f1),
        "ppv":               float(ppv),
        "npv":               float(npv),
        "confusion_matrix":  cm.tolist(),   # [[TN, FP], [FN, TP]]
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def compute_per_class_metrics(y_true, y_prob):
    """Per-class AUROC / AUC-PR for binary classification.
    Returns dict: {class_name: {precision, recall, f1, auroc, aucpr, n}}."""
    y_true = np.asarray(y_true)
    prob_pos = y_prob[:, 1]
    result = {}
    for cls_idx, cls_name in enumerate(['negative', 'positive']):
        cls_probs = prob_pos if cls_idx == 1 else 1 - prob_pos
        cls_labels = (y_true == cls_idx).astype(int)
        try:
            cls_auroc = float(roc_auc_score(cls_labels, cls_probs))
            cls_aucpr = float(average_precision_score(cls_labels, cls_probs))
        except Exception:
            cls_auroc = cls_aucpr = float('nan')
        result[cls_name] = {"auroc": cls_auroc, "aucpr": cls_aucpr}
    return result
