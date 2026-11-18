#!/usr/bin/env python3
"""EPIC-MobileNet 可迁移性门控: 决定新数据集上是否启用创新 ①(证据保持冻结 /2–/16)。

只用 train/val, 不读 test:
  LP_val = 冻结 ImageNet MobileNetV3-L 四尺度 GAP 证据 + 强 L2 逻辑回归(C 在 val 上选)的 val Macro-F1
  FT_val = 原协议全量微调 baseline(mobilenet_v3_large)各 seed 的 val Macro-F1 均值(读 evaluate.py 的 results.json)
  gap = FT_val − LP_val, 三档(门控 v2):
    gap ≤ 0            → 全保留: 不做阶段 1, 纯 ImageNet 主干 + 阶段 2(mbv3_lp_dce4m)
    0 < gap ≤ 阈值(1.5) → ① 冻结 /2–/16(阶段 1 mbv3_r6_fz16) + 阶段 2(mbv3_fz16_dce4m)
    gap > 阈值          → 全量微调(阶段 1 mobilenet_v3_large) + 阶段 2(mbv3_dce4m)
依据: LP-FT(Kumar et al., ICLR 2022) —— 线性探针已接近全量微调时, 微调主要带来特征失真。
两个阈值(0, 1.5)在 CLE/BUSI/heelBone/covid/CUB 上拟定, 并在 TCGA 上做了预注册样本外检验(见 创新说明.md §3)。

用法:
  python epic_gate.py --data datasets/cle_ac_v1 \
      --ft-runs weights/mobilenet_v3_large_seed42/mobilenet_v3_large weights/mobilenet_v3_large_seed43/mobilenet_v3_large
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torchvision import transforms

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from models.MobileNetV3 import mobilenet_v3_large  # noqa: E402

TAPS = (3, 6, 12, 16)
CGRID = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class _DS(torch.utils.data.Dataset):
    def __init__(self, root):
        cls = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        self.items = [(os.path.join(root, c, f), i) for i, c in enumerate(cls)
                      for f in sorted(os.listdir(os.path.join(root, c))) if f.lower().endswith(EXTS)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return TF(Image.open(p).convert("RGB")), y


@torch.no_grad()
def evidence(model, root, device, workers):
    acts = {}
    hooks = [model.features[i].register_forward_hook(lambda m, inp, o, k=i: acts.__setitem__(k, o)) for i in TAPS]
    X, Y = [], []
    for x, y in torch.utils.data.DataLoader(_DS(root), batch_size=128, num_workers=workers):
        model(x.to(device))
        X.append(torch.cat([acts[i].float().mean((2, 3)) for i in TAPS], 1).cpu())
        Y.append(y)
    for h in hooks:
        h.remove()
    return torch.cat(X).numpy(), torch.cat(Y).numpy()


def ft_val(run_dirs):
    vals = []
    for d in run_dirs:
        for f in sorted(glob.glob(os.path.join(d, "eval_*", "results.json")), reverse=True):
            r = json.load(open(f))
            if os.path.basename(r.get("test_dir", "").rstrip("/")) == "val":
                vals.append(r["macro_f1"] * 100)
                break
    if not vals:
        raise SystemExit("no val results.json found under --ft-runs (run evaluate.py on the val split first)")
    return float(np.mean(vals)), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root containing train/ and val/")
    ap.add_argument("--ft-runs", nargs="+", required=True, help="full fine-tuning baseline run dirs (with eval_*/results.json)")
    ap.add_argument("--pretrained", default="pth/mobilenet_v3_large-8738ca79.pth")
    ap.add_argument("--threshold", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = mobilenet_v3_large(num_classes=1000)
    m.load_state_dict(torch.load(a.pretrained, map_location="cpu"))
    m.eval().to(device)
    Xtr, ytr = evidence(m, os.path.join(a.data, "train"), device, a.workers)
    Xva, yva = evidence(m, os.path.join(a.data, "val"), device, a.workers)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    best = max(((f1_score(yva, LogisticRegression(C=C, max_iter=3000).fit((Xtr - mu) / sd, ytr).predict((Xva - mu) / sd),
                          average="macro") * 100, C) for C in CGRID), key=lambda t: t[0])
    ft, per = ft_val(a.ft_runs)
    gap = ft - best[0]
    if gap <= 0:
        decision = "preserve-all: no stage 1; stage 2 on the ImageNet backbone (--src-ckpt <pth> --dst-model mbv3_lp_dce4m)"
    elif gap <= a.threshold:
        decision = "enable ①: stage 1 mbv3_r6_fz16 -> stage 2 mbv3_fz16_dce4m"
    else:
        decision = "full fine-tuning: stage 1 mobilenet_v3_large -> stage 2 mbv3_dce4m"
    print(json.dumps({"LP_val_f1": round(best[0], 2), "LP_C": best[1], "FT_val_f1": round(ft, 2), "FT_val_per_run": per,
                      "gap": round(gap, 2), "thresholds": [0.0, a.threshold], "decision": decision},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
