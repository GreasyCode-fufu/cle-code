#!/usr/bin/env python3
"""R3 轮 cle_ac_v1 数据分布分析 (2026-11-08)

在 cle_image_analysis.py(11-01, 224 分辨率、每类采样 120 张) 之外补充原生分辨率与全量统计:
  Q1 划分/类别/亚型: 文件名前缀亚型与类别目录的对应, 样本 ID 跨划分重叠
  Q2 成像几何: 原生尺寸、视野形状(暗边/圆形 FOV)、强度分布(均值/对比度/饱和/暗像素)
  Q3 频谱: 原生分辨率径向功率谱; 800->256 缩放丢失的能量占比(256 Nyquist 以上频段)
  Q4 方向性: 梯度方向直方图各向异性指数(判断旋转不变性假设是否成立)
  Q5 泄漏: dHash 近重复(跨划分 Hamming<=4) 与 md5 完全重复
  Q6 类间可分性: 各低层特征的 one-vs-rest AUC(低层统计能否区分良/恶/正常)

输出: experiments/c_data_analysis.json + experiments/C_DATA_ANALYSIS.md
"""
import glob
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "datasets/cle_ac_v1")
OUT_JSON = os.path.join(REPO, "experiments", "c_data_analysis.json")
CLASSES = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]
SHORT = {"Benign_lesions": "B", "Malignant_lesions": "M", "body_of_stomach_normal": "N"}
NAME_RE = re.compile(r"^(train|val|test)_(.+?)_(S_[0-9A-Za-z]+)\.[^.]+$")
N_RBINS = 48          # 径向谱分箱(归一化到原生 Nyquist)
FFT_SIZE = 768        # 中心裁剪尺寸(原生约 800x800)


def radial_profile(g):
    """去均值 + Hann 窗后的径向平均功率谱, 返回 (N_RBINS,) 与按频段的能量占比."""
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2((g - g.mean()) * win))
    p = (f.real ** 2 + f.imag ** 2)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (min(h, w) / 2)  # 1.0 = 原生 Nyquist
    idx = np.clip((r * N_RBINS).astype(int), 0, N_RBINS)
    prof = np.bincount(idx.ravel(), weights=p.ravel(), minlength=N_RBINS + 1)[:N_RBINS]
    cnt = np.bincount(idx.ravel(), minlength=N_RBINS + 1)[:N_RBINS]
    prof = prof / np.maximum(cnt, 1)
    tot = p[r <= 1.0].sum() + 1e-9
    nyq256 = 256.0 / g.shape[0]          # 缩放到 256 后可保留的最高归一化频率
    nyq448 = 448.0 / g.shape[0]
    return prof, {
        "e_lost_256": float(p[(r > nyq256) & (r <= 1.0)].sum() / tot),
        "e_lost_448": float(p[(r > nyq448) & (r <= 1.0)].sum() / tot),
        "e_low": float(p[r <= 0.08].sum() / tot),
    }


def orientation_anisotropy(g):
    """梯度方向(倍角)二阶圆矩模长: 0=各向同性, 1=单一方向."""
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    th = np.arctan2(gy, gx)
    wsum = mag.sum() + 1e-9
    c = (mag * np.cos(2 * th)).sum() / wsum
    s = (mag * np.sin(2 * th)).sum() / wsum
    return float(np.hypot(c, s))


def dhash(im_gray, size=8):
    small = np.asarray(im_gray.resize((size + 1, size), Image.BILINEAR), dtype=np.float32)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def analyze(path):
    split, cls = path.split(os.sep)[-3], path.split(os.sep)[-2]
    m = NAME_RE.match(os.path.basename(path))
    subtype, sid = (m.group(2), m.group(3)) if m else ("?", "?")
    raw = open(path, "rb").read()
    im = Image.open(path)
    w, h = im.size
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    chan_diff = float(np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 0] - rgb[..., 2]).mean())
    g8 = im.convert("L")
    g = np.asarray(g8, dtype=np.float32) / 255.0
    # 视野: 暗像素占比, 四角 vs 中心亮度, 圆形 FOV 外环(内切圆外)亮度
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (min(h, w) / 2)
    outside = g[rr > 1.0]
    k = max(8, min(h, w) // 20)
    corners = np.concatenate([g[:k, :k].ravel(), g[:k, -k:].ravel(), g[-k:, :k].ravel(), g[-k:, -k:].ravel()])
    center = g[h // 2 - k:h // 2 + k, w // 2 - k:w // 2 + k]
    # 频谱: 中心裁剪 FFT_SIZE
    cy, cx = h // 2, w // 2
    s = min(FFT_SIZE, h, w)
    crop = g[cy - s // 2:cy - s // 2 + s, cx - s // 2:cx - s // 2 + s]
    prof, bands = radial_profile(crop)
    return {
        "split": split, "cls": cls, "subtype": subtype, "sid": sid, "w": w, "h": h,
        "md5": hashlib.md5(raw).hexdigest(), "dhash": dhash(g8),
        "chan_diff": chan_diff,
        "mean": float(g.mean()), "std": float(g.std()),
        "p01": float(np.percentile(g, 1)), "p99": float(np.percentile(g, 99)),
        "sat": float((g >= 250 / 255).mean()), "dark": float((g <= 10 / 255).mean()),
        "corner_mean": float(corners.mean()), "center_mean": float(center.mean()),
        "outside_circle_mean": float(outside.mean()) if outside.size else float("nan"),
        "aniso": orientation_anisotropy(crop[::2, ::2]),
        "prof": prof.tolist(), **bands,
    }


def auc_ovr(x, y_pos):
    """Mann-Whitney AUC, 返回 max(AUC, 1-AUC) 表示可分性强度."""
    x = np.asarray(x); y = np.asarray(y_pos, dtype=bool)
    pos, neg = x[y], x[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    a = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(max(a, 1 - a))


def main():
    paths = sorted(glob.glob(os.path.join(DATA, "*", "*", "*")))
    with Pool(8) as pool:
        rows = pool.map(analyze, paths, chunksize=16)
    rep = {"n_total": len(rows)}

    # Q1 划分/类别/亚型
    tab = Counter((r["split"], r["cls"], r["subtype"]) for r in rows)
    rep["subtype_table"] = {f"{s}|{c}|{t}": n for (s, c, t), n in sorted(tab.items())}
    sids = defaultdict(set)
    for r in rows:
        sids[r["split"]].add(r["sid"])
    rep["sid_overlap"] = {f"{a}&{b}": len(sids[a] & sids[b])
                          for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    rep["sid_unique_per_split"] = {k: len(v) for k, v in sids.items()}

    # Q2 几何/强度
    rep["size_wh_range"] = {"w": [min(r["w"] for r in rows), max(r["w"] for r in rows)],
                            "h": [min(r["h"] for r in rows), max(r["h"] for r in rows)]}
    keys = ["chan_diff", "mean", "std", "p01", "p99", "sat", "dark", "corner_mean", "center_mean",
            "outside_circle_mean", "aniso", "e_lost_256", "e_lost_448", "e_low"]
    per = {}
    for split in ["train", "val", "test"]:
        for c in CLASSES:
            rs = [r for r in rows if r["split"] == split and r["cls"] == c]
            per[f"{split}|{SHORT[c]}"] = {k: [round(float(np.nanmean([r[k] for r in rs])), 4),
                                              round(float(np.nanstd([r[k] for r in rs])), 4)] for k in keys}
            per[f"{split}|{SHORT[c]}"]["n"] = len(rs)
    rep["per_split_class"] = per

    # Q3 径向谱(train, 按类 log10 均值)
    rep["radial_log10_train"] = {
        SHORT[c]: np.log10(np.mean([r["prof"] for r in rows if r["split"] == "train" and r["cls"] == c],
                                   axis=0) + 1e-12).round(4).tolist() for c in CLASSES}

    # Q5 泄漏: md5 完全重复 / dHash 近重复
    md5_groups = defaultdict(list)
    for r in rows:
        md5_groups[r["md5"]].append((r["split"], r["cls"]))
    dup = [v for v in md5_groups.values() if len(v) > 1]
    rep["exact_dup_groups"] = len(dup)
    rep["exact_dup_cross_split"] = sum(1 for v in dup if len({s for s, _ in v}) > 1)
    rep["exact_dup_label_conflict"] = sum(1 for v in dup if len({c for _, c in v}) > 1)
    hashes = np.array([r["dhash"] for r in rows], dtype=np.uint64)
    split_arr = np.array([r["split"] for r in rows]); cls_arr = np.array([r["cls"] for r in rows])
    popcnt = np.array([bin(v).count("1") for v in range(256)], dtype=np.uint8)
    near = Counter(); near_conflict = Counter(); examples = []
    for i in range(len(rows)):
        x = np.bitwise_xor(hashes[i + 1:], hashes[i])
        d = popcnt[x.view(np.uint8)].reshape(-1, 8).sum(axis=1)
        for j in np.nonzero(d <= 4)[0]:
            j = i + 1 + j
            key = "&".join(sorted([split_arr[i], split_arr[j]]))
            near[key] += 1
            if cls_arr[i] != cls_arr[j]:
                near_conflict[key] += 1
            if split_arr[i] != split_arr[j] and len(examples) < 12:
                examples.append([os.path.relpath(paths[i], DATA), os.path.relpath(paths[j], DATA)])
    rep["near_dup_pairs_h4"] = dict(near)
    rep["near_dup_label_conflict_h4"] = dict(near_conflict)
    rep["near_dup_cross_split_examples"] = examples

    # Q6 低层特征可分性(train, one-vs-rest AUC)
    tr = [r for r in rows if r["split"] == "train"]
    feat_keys = ["mean", "std", "p01", "p99", "sat", "dark", "aniso", "e_lost_256", "e_low"]
    rep["ovr_auc_train"] = {k: {SHORT[c]: round(auc_ovr([r[k] for r in tr], [r["cls"] == c for r in tr]), 3)
                                for c in CLASSES} for k in feat_keys}
    bm = [r for r in tr if r["cls"] in ("Benign_lesions", "Malignant_lesions")]
    rep["auc_B_vs_M_train"] = {k: round(auc_ovr([r[k] for r in bm], [r["cls"] == "Malignant_lesions" for r in bm]), 3)
                               for k in feat_keys}

    json.dump(rep, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in ["n_total", "sid_overlap", "sid_unique_per_split", "size_wh_range",
                                          "exact_dup_groups", "exact_dup_cross_split", "exact_dup_label_conflict",
                                          "near_dup_pairs_h4", "near_dup_label_conflict_h4", "auc_B_vs_M_train"]},
                     ensure_ascii=False, indent=1))
    cols = ["mean", "std", "sat", "dark", "corner_mean", "outside_circle_mean", "aniso", "e_lost_256", "e_lost_448"]
    print(f"{'':10s}" + "".join(f"{c:>20s}" for c in cols))
    for k, v in per.items():
        print(f"{k:10s}" + "".join(f"{v[c][0]:>12.4f}±{v[c][1]:<7.4f}" for c in cols))
    print("radial log10 (train, every 6th bin):")
    for c, v in rep["radial_log10_train"].items():
        print(f"  {c}: " + " ".join(f"{x:6.2f}" for x in v[::6]))
    print("saved ->", OUT_JSON)


if __name__ == "__main__":
    main()
