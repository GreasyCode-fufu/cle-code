#!/usr/bin/env python3
"""CLE 图像物理特性分析 —— 数据驱动的网络设计依据 (2026-11-01)

三个问题:
  P1 类间物理可分性: 灰度冗余/亮度对比度/频谱能量/小波子带/边缘密度/纹理粗糙度
     -> 支撑创新点 B(频域边缘-结构重标定) 的设计动机
  P2 证据局部性: 局部方差图的 max/mean 与 top10% 能量占比 按类分组
     -> 支撑创新点 A(双池化多尺度头: GMP 保留最显著局部证据) 的动机
  P3 错误样本取证: v2bc seed42 test 的 M->B 漏诊样本 vs 正确样本 的物理特征差
     -> 直接定位新网络要解决的失效模式
  P4 下采样高频损失: 224->112 平均池化丢失的高频能量占比
     -> 支撑创新点 C(小波+MaxPool 混合下采样) 的动机

输出: experiments/cle_image_analysis.json + 控制台表格
"""
import csv
import glob
import json
import os
import random

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "datasets/cle_ac_v1")
OUT = os.path.join(REPO, "experiments", "cle_image_analysis.json")
CLASSES = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]
SHORT = {"Benign_lesions": "B", "Malignant_lesions": "M", "body_of_stomach_normal": "N"}
PRED = os.path.join(REPO, "weights_cle_ac_v1/convnext_v2bc_seed42/convnext_v2bc",
                    "eval_20261025_221600", "predictions.csv")
N_SAMPLE = 120
rng = random.Random(7)


def load_gray(path, size=224):
    im = Image.open(path).convert("RGB")
    im = im.resize((size, size), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return a, a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def radial_bins(g):
    """FFT 径向能量: 低(<=1/8半径)/中(1/8~1/3)/高(>1/3) 占比."""
    f = np.fft.fftshift(np.fft.fft2(g - g.mean()))
    p = (f.real ** 2 + f.imag ** 2)
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (max(h, w) / 2)
    tot = p.sum() + 1e-9
    return [float(p[r <= 1 / 8].sum() / tot), float(p[(r > 1 / 8) & (r <= 1 / 3)].sum() / tot),
            float(p[r > 1 / 3].sum() / tot)]


def haar_subbands(g):
    """2级Haar子带能量占比 [LL2, LH/HL/HH(细节总和)]."""
    a = g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2]
    lh = g[0::2, 0::2] + g[0::2, 1::2] - g[1::2, 0::2] - g[1::2, 1::2]
    hl = g[0::2, 0::2] - g[0::2, 1::2] + g[1::2, 0::2] - g[1::2, 1::2]
    hh = g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]
    e = [np.mean(a ** 2), np.mean(lh ** 2), np.mean(hl ** 2), np.mean(hh ** 2)]
    det = sum(e[1:]) / (sum(e) + 1e-9)
    return float(det)


def sobel_mag(g):
    gx = np.hypot(np.delete(np.diff(g, axis=1), -1, 1)[:, :-1], 0) if False else None
    kx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    ky = kx.T.copy()
    from scipy_free_conv import conv2  # placeholder replaced below
    return None


def edge_density(g):
    """无scipy: 手写Sobel."""
    kx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = conv3(g, kx)
    gy = conv3(g, kx.T.copy())
    return float(np.hypot(gx, gy).mean())


def conv3(g, k):
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(g, (3, 3))
    return (w * k).sum(axis=(-2, -1))


def local_stats(g, cell=14):
    """16x16网格局部std: 返回 (mean_std, max_std, top10能量占比, gini)."""
    h = g.reshape(16, cell, 16, cell).transpose(0, 2, 1, 3).reshape(16, 16, -1)
    s = h.std(axis=-1).ravel() + 1e-6
    ms = float(s.mean())
    order = np.sort(s)[::-1]
    top10 = float(order[: max(1, len(order) // 10)].sum() / (s.sum() + 1e-9))
    n = len(s)
    gini = float(np.abs(s[:, None] - s[None, :]).sum() / (2 * n * n * s.mean()))
    return ms, float(s.max()), top10, gini


def feats(path):
    a, g = load_gray(path)
    r, gg, b = a[..., 0], a[..., 1], a[..., 2]
    gray_redund = float(np.abs(r - gg).mean() + np.abs(r - b).mean())
    lo, mid, hi = radial_bins(g)
    ms, mx, t10, gi = local_stats(g)
    # P4: 224->112 avgpool 再 upsample 回来的能量损失(高频丢失)
    small = g.reshape(112, 2, 112, 2).mean(axis=(1, 3))
    up = np.kron(small, np.ones((2, 2), dtype=np.float32))
    hf_loss = float(((g - up) ** 2).mean() / ((g - g.mean()) ** 2).mean())
    return {
        "gray_redund": gray_redund, "brightness": float(g.mean()), "contrast": float(g.std()),
        "fft_lo": lo, "fft_mid": mid, "fft_hi": hi, "haar_det": haar_subbands(g),
        "edge": edge_density(g), "loc_std_mean": ms, "loc_std_max": mx,
        "loc_top10": t10, "loc_gini": gi, "max_over_mean": mx / (ms + 1e-9),
        "ds_hf_loss": hf_loss,
    }


def agg(rows):
    keys = rows[0].keys()
    return {k: [round(float(np.mean([r[k] for r in rows])), 4),
                round(float(np.std([r[k] for r in rows])), 4)] for k in keys}


def main():
    report = {}
    # P1/P2: train 每类采样
    per_class = {}
    for c in CLASSES:
        fs = sorted(glob.glob(os.path.join(DATA, "train", c, "*")))
        fs = rng.sample(fs, min(N_SAMPLE, len(fs)))
        rows = [feats(f) for f in fs]
        per_class[c] = agg(rows)
    report["per_class"] = per_class

    # P3: 错误取证
    errs = {"M->B": [], "M_ok": [], "B->M": []}
    with open(PRED) as f:
        for row in csv.DictReader(f):
            t, p = row["true_label"], row["pred_label"]
            key = None
            if t == "Malignant_lesions" and p == "Benign_lesions":
                key = "M->B"
            elif t == "Malignant_lesions" and p == t:
                key = "M_ok" if len(errs["M_ok"]) < 40 else None
            elif t == "Benign_lesions" and p == "Malignant_lesions":
                key = "B->M"
            if key:
                errs[key].append(os.path.join(REPO, row["image_path"].lstrip("./")))
    report["n_err"] = {k: len(v) for k, v in errs.items()}
    for k, v in errs.items():
        if v:
            report[f"err_{k}"] = agg([feats(f) for f in v])

    json.dump(report, open(OUT, "w"), indent=1, ensure_ascii=False)

    # 控制台表
    cols = ["brightness", "contrast", "fft_hi", "haar_det", "edge",
            "max_over_mean", "loc_top10", "loc_gini", "ds_hf_loss", "gray_redund"]
    print(f"{'':14s}" + "".join(f"{c:>12s}" for c in cols))
    for c in CLASSES:
        m = per_class[c]
        print(f"{SHORT[c]:14s}" + "".join(f"{m[k][0]:>12.4f}" for k in cols))
    for k in ["M->B", "M_ok", "B->M"]:
        if f"err_{k}" in report:
            m = report[f"err_{k}"]
            print(f"{k:14s}" + "".join(f"{m[k2][0]:>12.4f}" for k2 in cols))
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
