#!/usr/bin/env python3
"""DPFusion inference: decision-level fusion of the LCN and SN prior branches.

The final CLE model of this project is not a single forward pass: it averages the
softmax outputs of two identical ConvNeXt-T backbones whose *only* difference is
the imaging prior attached to the input (local contrast normalisation vs.
spectral notch).  This script reproduces that system end-to-end on the test split
and prints every metric used in the paper.

Usage:
  python scripts/predict_fusion.py \
      --lcn-weights weights_main/cle_convnext_lcn_only-1/model_best.pth \
      --sn-weights  weights_main/cle_convnext_sn/model_best.pth \
      --data-path   ./datasets/cle_ac_v1/test
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CLASSES = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(model_name: str, weights: str, device):
    import cle_fair_common as c
    model = c.build_model(model_name, len(CLASSES))
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def list_test(data_path: str):
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        d = os.path.join(data_path, cls)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                paths.append(os.path.join(d, fn))
                labels.append(idx)
    return paths, np.array(labels)


@torch.no_grad()
def infer(model, paths, tf, device, batch=64):
    out = []
    for i in range(0, len(paths), batch):
        imgs = [tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]
        x = torch.stack(imgs).to(device)
        out.append(torch.softmax(model(x).float(), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcn-weights", required=True)
    ap.add_argument("--sn-weights", required=True)
    ap.add_argument("--data-path", default="./datasets/cle_ac_v1/test")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out-dir", default="experiments/cle_agent/fusion_eval")
    args = ap.parse_args()

    from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score,
                                 f1_score, matthews_corrcoef, precision_score, recall_score,
                                 roc_auc_score)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.143)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    paths, y_true = list_test(os.path.join(ROOT, args.data_path) if not os.path.isabs(args.data_path)
                              else args.data_path)
    print(f"[data] {len(paths)} test images")

    m_lcn = load_model("cle_convnext_lcn_only", os.path.join(ROOT, args.lcn_weights), device)
    m_sn = load_model("cle_convnext_sn", os.path.join(ROOT, args.sn_weights), device)

    p_lcn = infer(m_lcn, paths, tf, device)
    p_sn = infer(m_sn, paths, tf, device)
    prob = 0.5 * (p_lcn + p_sn)
    y_pred = prob.argmax(1)

    spec = []
    for c in range(len(CLASSES)):
        tn = int(np.sum((y_true != c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)

    m = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_sensitivity": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_specificity": float(np.mean(spec)),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "auc_roc_macro": roc_auc_score(y_true, prob, multi_class="ovr", average="macro"),
        "auc_pr_macro": float(np.mean([average_precision_score((y_true == c).astype(int), prob[:, c])
                                       for c in range(len(CLASSES))])),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "n_errors": int((y_true != y_pred).sum()),
    }

    print("\n=== DPFusion (LCN + SN, decision-level) ===")
    for k, v in m.items():
        print(f"  {k:20s} {v:.4f}" if isinstance(v, float) else f"  {k:20s} {v}")

    os.makedirs(os.path.join(ROOT, args.out_dir), exist_ok=True)
    with open(os.path.join(ROOT, args.out_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "true_label", "pred_label"] + [f"prob_{c}" for c in CLASSES]
                   + ["prob_lcn_argmax", "prob_sn_argmax"])
        for i, p in enumerate(paths):
            w.writerow([p, CLASSES[y_true[i]], CLASSES[y_pred[i]]] +
                       [f"{v:.6f}" for v in prob[i]] +
                       [CLASSES[int(p_lcn[i].argmax())], CLASSES[int(p_sn[i].argmax())]])
    print(f"\n[saved] {args.out_dir}/predictions.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
