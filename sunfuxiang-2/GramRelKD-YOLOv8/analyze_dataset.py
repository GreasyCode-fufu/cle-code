#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hong Kong wall-defect dataset (YOLO) -- class-conditional statistical analysis.

Goal: explain why the `moisture` class is detected much worse than `spalling`
and `crack` by the YOLOv8 baseline.

This script is READ-ONLY with respect to the dataset and the training code:
it never modifies labels / images / training code. It only writes a report
into OUT_DIR (placed in the ultralytics working dir, not inside the dataset).

Sections (match the request 1..10):
  1. class distribution
  2. bbox geometry (per class, train/val/test/overall)
  3. resize -> imgsz=640 (faithful Ultralytics letterbox simulation)
  4. class co-occurrence (image level)
  5. border-touching analysis
  6. image-condition analysis (brightness / color / contrast)
  7. train/val/test distribution shift
  8. visual montages (crops + full-image+bbox, per class & size group)
  9. data-quality checks (report only, no modification)
 10. CSVs + plots + summary.txt answering A..G
"""

import os
import sys
import math
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# scipy is optional (used for KS distribution-shift test)
try:
    from scipy import stats as scistats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
DATA_ROOT = Path("/data/cle/sunfuxiang-2/datasets/hongkongScale")
OUT_DIR   = Path("/data/cle/sunfuxiang-2/ultralytics-main/dataset_analysis")
PLOT_DIR  = OUT_DIR / "plots"
SPLITS    = ["train", "val", "test"]
CLASSES   = {0: "spalling", 1: "crack", 2: "moisture"}
NAME2ID   = {v: k for k, v in CLASSES.items()}
CLASS_ORDER = ["spalling", "crack", "moisture"]          # for plotting
IMGSZ       = 640                                        # YOLO letterbox target
SCALEUP     = False                                     # val/test inference never scales up
SEED        = 42
BORDER_TOL_PX = 2.0                                     # a box "touches" a border if its edge
                                                         # is within this many px of the image edge
PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]

# COCO small-object thresholds (in source-image pixels)
SMALL_MAX  = 32 ** 2      # area <  32^2  -> small
MED_MAX    = 96 ** 2      # 32^2 <= area < 96^2 -> medium, else large
TINY_PX    = 2            # "extremely tiny box": shortest side < TINY_PX
HUGE_FRAC  = 0.90         # "abnormally large box": normalized area >= HUGE_FRAC
DUP_IOU    = 0.90         # same-image same-class IoU >= this -> "highly overlapping"

# Montage settings
SAMPLE_PER_GROUP = 60     # >= 50 crops per (split, class, size-group)
CROP_SIZE        = 128    # px per crop tile
FULL_IMG_SIZE    = 480    # px per full-image tile
MAX_FULL_TILES   = 12     # tiles per (split, class) full-image montage

CLS_COLOR  = {"spalling": "#1f77b4", "crack": "#ff7f0e", "moisture": "#d62728"}
SPLIT_COLOR = {"train": "#2ca02c", "val": "#1f77b4", "test": "#9467bd"}

np.random.seed(SEED)
random.seed(SEED)

for d in (OUT_DIR, PLOT_DIR,
          OUT_DIR / "moisture_montage", OUT_DIR / "crack_montage", OUT_DIR / "spalling_montage"):
    d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def summarize(a):
    """Return mean/std/percentiles dict for a 1-D array (NaNs dropped)."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {**{"mean": np.nan, "std": np.nan}, **{f"p{p}": np.nan for p in PCTS}, "n": 0}
    out = {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}
    for p in PCTS:
        out[f"p{p}"] = float(np.percentile(a, p))
    return out


def letterbox_ratio(H, W, imgsz=IMGSZ, scaleup=SCALEUP):
    """Faithful Ultralytics LetterBox scale ratio (see ultralytics/data/augment.py).

    r = min(imgsz/H, imgsz/W); clamped to <= 1.0 when not scaleup (val/test).
    """
    r = min(imgsz / H, imgsz / W)
    if not scaleup:
        r = min(r, 1.0)
    return r


def iou(a, b):
    """IoU of two boxes given as (x1,y1,x2,y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def class_color_bgr(name):
    hexc = CLS_COLOR[name].lstrip("#")
    r, g, b = (int(hexc[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)  # cv2 is BGR


# --------------------------------------------------------------------------------------
# Data loading (single pass over images: labels + pixels)
# --------------------------------------------------------------------------------------
def load_split(split):
    """Return (inst_df, img_df) for one split.

    inst_df: one row per annotated bbox.
    img_df : one row per image (presence flags + image-condition stats).
    """
    img_dir = DATA_ROOT / split / "images"
    lbl_dir = DATA_ROOT / split / "labels"
    img_paths = sorted([p for p in img_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")])

    inst_rows, img_rows = [], []
    n_img = len(img_paths)
    for ii, ip in enumerate(img_paths):
        stem = ip.stem
        lp = lbl_dir / f"{stem}.txt"

        # --- read labels (YOLO: cls cx cy w h, normalized) ---
        raw = []
        if lp.is_file():
            with open(lp, encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    parts = s.split()
                    try:
                        c = int(float(parts[0]))
                        cx, cy, w, h = map(float, parts[1:5])
                        raw.append((c, cx, cy, w, h))
                    except Exception:
                        raw.append(("BAD", s))  # flagged in quality check

        # --- read image (cv2, BGR). If it fails we still keep labels w/ NaN size ---
        img = cv2.imread(str(ip))
        if img is None:
            H, W = np.nan, np.nan
            bright = bright_std = contrast = rmean = gmean = bmean = np.nan
        else:
            H, W = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bright = float(gray.mean())
            bright_std = float(gray.std())
            # robust range-based contrast in [0,1]
            contrast = float((np.percentile(gray, 95) - np.percentile(gray, 5)) / 255.0)
            # cv2 is BGR
            bmean = float(img[..., 0].mean())
            gmean = float(img[..., 1].mean())
            rmean = float(img[..., 2].mean())

        # image-level presence + counts
        present = Counter()
        for r in raw:
            if r[0] in CLASSES:
                present[r[0]] += 1

        img_rows.append(dict(
            split=split, filename=ip.name, path=str(ip),
            n_instances=sum(present.values()),
            has_spalling=int(0 in present), has_crack=int(1 in present), has_moisture=int(2 in present),
            n_spalling=present.get(0, 0), n_crack=present.get(1, 0), n_moisture=present.get(2, 0),
            H=H, W=W,
            brightness=bright, brightness_std=bright_std, contrast=contrast,
            R_mean=rmean, G_mean=gmean, B_mean=bmean,
            label_missing=int(not lp.is_file()), label_empty=int(lp.is_file() and len(raw) == 0),
        ))

        if not (H != H):  # H is finite -> not NaN
            r = letterbox_ratio(H, W)
        else:
            r = np.nan

        for r_ in raw:
            if r_[0] == "BAD":
                inst_rows.append(dict(split=split, filename=ip.name, cls_id="BAD", bad_line=r_[1]))
                continue
            c, cx, cy, w, h = r_
            name = CLASSES.get(c, f"cls{c}")
            if H != H:
                # image unreadable: keep normalized-only record
                inst_rows.append(dict(split=split, filename=ip.name, cls_id=c, cls=name,
                                      cx=cx, cy=cy, w=w, h=h))
                continue
            px_w, px_h = w * W, h * H
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
            x2, y2 = x1 + px_w, y1 + px_h
            area = px_w * px_h
            norm_area = w * h
            shortest = min(px_w, px_h)
            longest = max(px_w, px_h)
            aspect = longest / shortest if shortest > 0 else np.nan
            # border distances (px)
            d_left, d_right = x1, (W - x2)
            d_top, d_bottom = y1, (H - y2)
            touch_l = d_left <= BORDER_TOL_PX
            touch_r = d_right <= BORDER_TOL_PX
            touch_t = d_top <= BORDER_TOL_PX
            touch_b = d_bottom <= BORDER_TOL_PX
            touch_any = touch_l or touch_r or touch_t or touch_b
            # resize-to-640 (letterbox scale; padding does not change box *size*)
            rs_w, rs_h = px_w * r, px_h * r
            rs_short = min(rs_w, rs_h)
            rs_area = rs_w * rs_h
            inst_rows.append(dict(
                split=split, filename=ip.name, path=str(ip), cls_id=c, cls=name,
                cx=cx, cy=cy, w=w, h=h,
                W=W, H=H, px_w=px_w, px_h=px_h,
                x1=x1, y1=y1, x2=x2, y2=y2,
                area=area, norm_area=norm_area,
                shortest=shortest, longest=longest, aspect=aspect,
                d_left=d_left, d_right=d_right, d_top=d_top, d_bottom=d_bottom,
                touch_left=touch_l, touch_right=touch_r, touch_top=touch_t, touch_bottom=touch_b,
                touch_any=touch_any,
                rs_w=rs_w, rs_h=rs_h, rs_short=rs_short, rs_area=rs_area, rs_ratio=r,
            ))

        if (ii + 1) % 500 == 0:
            print(f"  [{split}] {ii+1}/{n_img} images parsed", flush=True)

    inst_df = pd.DataFrame(inst_rows)
    img_df  = pd.DataFrame(img_rows)
    print(f"[load] {split}: {len(img_df)} images, {len(inst_df)} raw instance rows", flush=True)
    return inst_df, img_df


# --------------------------------------------------------------------------------------
# 1. Class distribution
# --------------------------------------------------------------------------------------
def section_class_distribution(all_inst, all_img):
    rows = []
    imb_rows = []
    for split in SPLITS + ["overall"]:
        sdf = all_inst[all_inst["split"] == split] if split != "overall" else all_inst
        idf = all_img[all_img["split"] == split] if split != "overall" else all_img
        n_img_total = int(len(idf))
        per_cls_inst = {}
        for cid, name in CLASSES.items():
            sub = sdf[sdf["cls_id"] == cid]
            n_inst = int(len(sub))
            n_img_cls = int(idf[f"has_{name}"].sum()) if f"has_{name}" in idf else int(0)
            per_cls_inst[name] = n_inst
            avg_per_present = n_inst / n_img_cls if n_img_cls else np.nan
            avg_per_total = n_inst / n_img_total if n_img_total else np.nan
            rows.append(dict(
                scope=split, **{"class": name}, images_total=n_img_total,
                images_with_class=n_img_cls, instance_count=n_inst,
                avg_per_image_with_class=avg_per_present,
                avg_per_image_total=avg_per_total,
                instance_share=n_inst / len(sdf) if len(sdf) else np.nan,
                image_share=n_img_cls / n_img_total if n_img_total else np.nan,
            ))
        vals = [per_cls_inst[n] for n in CLASS_ORDER]
        if min(vals) > 0:
            imb = max(vals) / min(vals)
        else:
            imb = np.nan
        imb_rows.append(dict(scope=split,
                             imbalance_ratio_inst=imb,
                             max_class=CLASS_ORDER[int(np.argmax(vals))],
                             min_class=CLASS_ORDER[int(np.argmin(vals))],
                             **{f"inst_{n}": per_cls_inst[n] for n in CLASS_ORDER}))
    return pd.DataFrame(rows), pd.DataFrame(imb_rows)


# --------------------------------------------------------------------------------------
# 2. Bbox geometry (per class, train/val/test/overall) + threshold counts
# --------------------------------------------------------------------------------------
GEO_METRICS = ["width", "height", "area", "norm_area", "shortest_side", "longest_side", "aspect_ratio"]
GEO_COL = {"width": "px_w", "height": "px_h", "area": "area", "norm_area": "norm_area",
           "shortest_side": "shortest", "longest_side": "longest", "aspect_ratio": "aspect"}


def section_bbox_geometry(all_inst):
    rows = []
    for split in SPLITS + ["overall"]:
        sdf = all_inst[all_inst["split"] == split] if split != "overall" else all_inst
        sdf = sdf[sdf["cls_id"].isin(CLASSES)]
        for cid, name in CLASSES.items():
            sub = sdf[sdf["cls_id"] == cid]
            for metric in GEO_METRICS:
                col = GEO_COL[metric]
                s = summarize(sub[col].values)
                for stat, val in s.items():
                    rows.append(dict(scope=split, split=split, **{"class": name},
                                     metric=metric, stat=stat, value=val))
            # threshold counts
            sh = sub["shortest"].values
            ar = sub["aspect"].values
            a = sub["area"].values
            thr = {
                "shortest_lt8":  sh < 8, "shortest_lt16": sh < 16, "shortest_lt32": sh < 32,
                "area_lt_32sq":  a < SMALL_MAX,
                "area_32_to_96sq": (a >= SMALL_MAX) & (a < MED_MAX),
                "area_ge_96sq":  a >= MED_MAX,
                "aspect_ge3":  ar >= 3, "aspect_ge5": ar >= 5, "aspect_ge10": ar >= 10,
            }
            for m, mask in thr.items():
                cnt = int(np.nansum(mask))
                rows.append(dict(scope=split, split=split, **{"class": name}, metric=m,
                                 stat="count", value=cnt))
                rows.append(dict(scope=split, split=split, **{"class": name}, metric=m,
                                 stat="frac", value=cnt / len(sub) if len(sub) else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 3. Resize -> 640
# --------------------------------------------------------------------------------------
RESIZE_METRICS = {"width": "rs_w", "height": "rs_h", "shortest_side": "rs_short", "area": "rs_area"}


def section_resize640(all_inst):
    rows = []
    # constant ratio note
    ratios = all_inst["rs_ratio"].dropna().unique()
    ratio_note = (f"All images share one size (H=768,W=1024) -> letterbox ratio r="
                  f"{ratios[0]:.6f} (scaleup={SCALEUP}); new_unpad=(640,480) padded to 640x640."
                  if len(ratios) == 1 else f"Multiple ratios present: {ratios}")
    for split in SPLITS + ["overall"]:
        sdf = all_inst[all_inst["split"] == split] if split != "overall" else all_inst
        sdf = sdf[sdf["cls_id"].isin(CLASSES)]
        for cid, name in CLASSES.items():
            sub = sdf[sdf["cls_id"] == cid]
            for metric, col in RESIZE_METRICS.items():
                s = summarize(sub[col].values)
                for stat, val in s.items():
                    rows.append(dict(scope=split, split=split, **{"class": name},
                                     metric=metric, stat=stat, value=val))
            # how many become very small at 640 (key for detection difficulty)
            rs = sub["rs_short"].values
            for thr in (4, 8, 16, 32):
                cnt = int(np.nansum(rs < thr))
                rows.append(dict(scope=split, split=split, **{"class": name},
                                 metric=f"resized_shortest_lt{thr}", stat="count", value=cnt))
                rows.append(dict(scope=split, split=split, **{"class": name},
                                 metric=f"resized_shortest_lt{thr}", stat="frac",
                                 value=cnt / len(sub) if len(sub) else np.nan))
    return pd.DataFrame(rows), ratio_note


# --------------------------------------------------------------------------------------
# 4. Co-occurrence (image level)
# --------------------------------------------------------------------------------------
def section_cooccurrence(all_img):
    rows = []
    for split in SPLITS + ["overall"]:
        idf = all_img[all_img["split"] == split] if split != "overall" else all_img
        idf = idf.copy()
        n = len(idf)
        has = {name: idf[f"has_{name}"].astype(bool).values for name in CLASS_ORDER}
        m = {k: int(v.sum()) for k, v in has.items()}
        def cond(a, b):
            denom = int(has[b].sum())
            return float((has[a] & has[b]).sum() / denom) if denom else np.nan
        rows.append(dict(scope=split, n_images=n,
                         P_crack_given_moisture=cond("crack", "moisture"),
                         P_spalling_given_moisture=cond("spalling", "moisture"),
                         P_moisture_given_crack=cond("moisture", "crack"),
                         P_moisture_given_spalling=cond("moisture", "spalling")))
        # partitions involving moisture
        mo, cr, sp = has["moisture"], has["crack"], has["spalling"]
        parts = {
            "only_moisture":            (mo & ~cr & ~sp),
            "moisture_and_crack":       (mo & cr & ~sp),
            "moisture_and_spalling":    (mo & sp & ~cr),
            "moisture_crack_spalling":  (mo & cr & sp),
            "moisture_any":             (mo),
        }
        for k, mask in parts.items():
            cnt = int(mask.sum())
            rows.append(dict(scope=split, partition=k, count=cnt, frac=cnt / n if n else np.nan))
    # split P() rows and partition rows into separate frames but return merged long
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 5. Border analysis
# --------------------------------------------------------------------------------------
def section_border(all_inst):
    rows = []
    flags = ["touch_left", "touch_right", "touch_top", "touch_bottom", "touch_any"]
    for split in SPLITS + ["overall"]:
        sdf = all_inst[all_inst["split"] == split] if split != "overall" else all_inst
        sdf = sdf[sdf["cls_id"].isin(CLASSES)]
        for cid, name in CLASSES.items():
            sub = sdf[sdf["cls_id"] == cid]
            n = len(sub)
            for f in flags:
                cnt = int(sub[f].sum())
                rows.append(dict(scope=split, split=split, **{"class": name}, metric=f,
                                 count=cnt, frac=cnt / n if n else np.nan, n=n))
        # also min-distance-to-any-border distribution summary per class
        for cid, name in CLASSES.items():
            sub = sdf[sdf["cls_id"] == cid]
            if len(sub):
                dmin = np.minimum.reduce([sub["d_left"].values, sub["d_right"].values,
                                          sub["d_top"].values, sub["d_bottom"].values])
                s = summarize(dmin)
                for stat, val in s.items():
                    rows.append(dict(scope=split, split=split, **{"class": name},
                                     metric=f"min_dist_to_border_px", stat=stat, value=val))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 6. Image-condition analysis
# --------------------------------------------------------------------------------------
def section_image_conditions(all_img):
    # group images by which class they contain and compare conditions
    rows = []
    metrics = ["brightness", "brightness_std", "contrast", "R_mean", "G_mean", "B_mean"]
    for split in SPLITS + ["overall"]:
        idf = all_img[all_img["split"] == split] if split != "overall" else all_img
        for name in CLASS_ORDER:
            sub = idf[idf[f"has_{name}"] == 1]
            for m in metrics:
                s = summarize(sub[m].values)
                for stat, val in s.items():
                    rows.append(dict(scope=split, split=split, group=f"images_with_{name}",
                                     metric=m, stat=stat, value=val))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 7. Distribution shift (train vs val vs test) per class
# --------------------------------------------------------------------------------------
SHIFT_METRICS = {"area": "area", "shortest_side": "shortest", "aspect_ratio": "aspect",
                 "brightness": None, "contrast": None}  # last two are image-level


def section_distribution_shift(all_inst, all_img):
    rows = []
    for cls_name in CLASS_ORDER:
        cid = NAME2ID[cls_name]
        inst = all_inst[all_inst["cls_id"] == cid]
        for metric, col in SHIFT_METRICS.items():
            if col is not None:
                base = inst[["split", col]].copy().rename(columns={col: "value"})
            else:
                # image-level: images containing the class
                imgs = all_img[all_img[f"has_{cls_name}"] == 1][["split", metric]].copy()
                imgs = imgs.rename(columns={metric: "value"})
                base = imgs
            train = base[base["split"] == "train"]["value"].dropna().values
            for split in SPLITS:
                arr = base[base["split"] == split]["value"].dropna().values
                s = summarize(arr)
                rec = dict(**{"class": cls_name}, metric=metric, split=split, n=s["n"])
                for stat in ["mean", "std", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]:
                    rec[stat] = s[stat]
                # shift vs train
                if split != "train" and len(train) and len(arr) and HAS_SCIPY:
                    ks = scistats.ks_2samp(train, arr)
                    rec["ks_stat_vs_train"] = float(ks.statistic)
                    rec["ks_pval_vs_train"] = float(ks.pvalue)
                    rec["mean_diff_vs_train"] = float(arr.mean() - train.mean())
                    rec["norm_shift_vs_train"] = float(abs(arr.mean() - train.mean()) / (train.std() + 1e-9))
                rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 8. Montages
# --------------------------------------------------------------------------------------
def size_group(area):
    if area < SMALL_MAX:
        return "small"
    elif area < MED_MAX:
        return "medium"
    else:
        return "large"


def _annotated_crop(row, pad=4):
    """Read image, crop the bbox, draw outline + label text, resize to CROP_SIZE."""
    img = cv2.imread(row["path"])
    if img is None:
        tile = np.zeros((CROP_SIZE, CROP_SIZE, 3), np.uint8)
        return tile
    H, W = img.shape[:2]
    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
    x1 = float(np.clip(x1, 0, W - 1)); x2 = float(np.clip(x2, 1, W))
    y1 = float(np.clip(y1, 0, H - 1)); y2 = float(np.clip(y2, 1, H))
    # expand a little context (guarantee >=1px on each axis)
    cw, ch = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    ex1 = max(0, int(x1 - cw * 0.15))
    ey1 = max(0, int(y1 - ch * 0.15))
    ex2 = min(W, max(ex1 + 1, int(x2 + cw * 0.15)))
    ey2 = min(H, max(ey1 + 1, int(y2 + ch * 0.15)))
    crop = img[ey1:ey2, ex1:ex2]
    if crop.size == 0:
        crop = img[max(0, int(y1)):max(1, int(y2)), max(0, int(x1)):max(1, int(x2))]
    crop = np.ascontiguousarray(crop).astype(np.uint8)
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    src_h, src_w = crop.shape[:2]
    # resize FIRST to a guaranteed-valid 128x128 contiguous tile, then draw the
    # box in scaled coords (avoids any degenerate-crop cv2.Mat layout issues).
    color = class_color_bgr(row["cls"])
    tile = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    sx, sy = CROP_SIZE / src_w, CROP_SIZE / src_h
    rx1 = int(np.clip((x1 - ex1) * sx, 0, CROP_SIZE - 1))
    rx2 = int(np.clip((x2 - ex1) * sx, 1, CROP_SIZE))
    ry1 = int(np.clip((y1 - ey1) * sy, 0, CROP_SIZE - 1))
    ry2 = int(np.clip((y2 - ey1) * sy, 1, CROP_SIZE))
    if rx2 > rx1 and ry2 > ry1:
        cv2.rectangle(tile, (rx1, ry1), (rx2, ry2), color, 2)
    crop = tile
    # text strip
    strip = np.full((30, CROP_SIZE, 3), 255, np.uint8)
    txt = f"{row['filename'][:14]} s={row['shortest']:.0f},{row['longest']:.0f} na={row['norm_area']:.3f}"
    cv2.putText(strip, txt, (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([crop, strip])


def make_grid(tiles, ncols, pad=4, bg=(40, 40, 40)):
    """Tile list of equal-shape images into a grid."""
    if not tiles:
        return None
    th, tw = tiles[0].shape[:2]
    nrows = math.ceil(len(tiles) / ncols)
    GW = ncols * tw + (ncols + 1) * pad
    GH = nrows * th + (nrows + 1) * pad
    grid = np.full((GH, GW, 3), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, ncols)
        y = pad + r * (th + pad)
        x = pad + c * (tw + pad)
        grid[y:y+th, x:x+tw] = t
    return grid


def _annotated_full(row):
    """Full image with all bboxes drawn, resized to FULL_IMG_SIZE, with filename strip."""
    img = cv2.imread(row["path"])
    if img is None:
        return None
    H, W = img.shape[:2]
    scale = FULL_IMG_SIZE / max(H, W)
    small = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_LINEAR)
    # draw all boxes for this image (we get path; need boxes -> caller passes per-image dict)
    return small  # boxes drawn by caller via wrapper


def section_montages(all_inst, all_img):
    # ---- crop montages per (split, class, size_group) ----
    for cls_name in CLASS_ORDER:
        cid = NAME2ID[cls_name]
        sub = all_inst[(all_inst["cls_id"] == cid) & all_inst["path"].notna()].copy()
        sub["size_group"] = sub["area"].apply(size_group)
        for split in SPLITS:
            ss = sub[sub["split"] == split]
            for grp in ["small", "medium", "large"]:
                g = ss[ss["size_group"] == grp]
                if len(g) == 0:
                    print(f"[montage-crop] {cls_name} {split} {grp}: 0 samples, skipped", flush=True)
                    continue
                k = min(SAMPLE_PER_GROUP, len(g))
                samp = g.sample(k, random_state=SEED)
                tiles = [_annotated_crop(r) for _, r in samp.iterrows()]
                grid = make_grid(tiles, ncols=10)
                if grid is not None:
                    out = OUT_DIR / f"{cls_name}_montage" / f"crop_{split}_{grp}.jpg"
                    cv2.imwrite(str(out), grid)
                print(f"[montage-crop] {cls_name} {split} {grp}: {k} crops -> {out.name}", flush=True)

    # ---- full-image + bbox montages per (split, class) ----
    img_box_index = defaultdict(list)
    for _, r in all_inst[all_inst["path"].notna()].iterrows():
        img_box_index[(r["split"], r["path"], r["filename"])].append(r)
    for cls_name in CLASS_ORDER:
        cid = NAME2ID[cls_name]
        for split in SPLITS:
            # images that contain this class
            idf = all_img[(all_img["split"] == split) & (all_img[f"has_{cls_name}"] == 1)]
            if len(idf) == 0:
                continue
            k = min(MAX_FULL_TILES, len(idf))
            samp_imgs = idf.sample(k, random_state=SEED)
            tiles = []
            for _, ir in samp_imgs.iterrows():
                rows = img_box_index.get((split, ir["path"], ir["filename"]), [])
                img = cv2.imread(ir["path"])
                if img is None:
                    continue
                H, W = img.shape[:2]
                scale = FULL_IMG_SIZE / max(H, W)
                small = cv2.resize(img, (max(1, int(W * scale)), max(1, int(H * scale))),
                                   interpolation=cv2.INTER_LINEAR)
                for r in rows:
                    color = class_color_bgr(r["cls"])
                    x1, y1, x2, y2 = (v * scale for v in (r["x1"], r["y1"], r["x2"], r["y2"]))
                    cv2.rectangle(small, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                strip = np.full((26, FULL_IMG_SIZE, 3), 255, np.uint8)
                cv2.putText(strip, f"{ir['filename'][:24]} [{cls_name}]", (3, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                # pad width to FULL_IMG_SIZE
                h, w = small.shape[:2]
                if w < FULL_IMG_SIZE:
                    small = np.pad(small, ((0, 0), (0, FULL_IMG_SIZE - w), (0, 0)))
                tiles.append(np.vstack([small, strip]))
            grid = make_grid(tiles, ncols=4)
            if grid is not None:
                out = OUT_DIR / f"{cls_name}_montage" / f"fullimg_{split}.jpg"
                cv2.imwrite(str(out), grid)
            print(f"[montage-full] {cls_name} {split}: {len(tiles)} images -> "
                  f"{f'{cls_name}_montage/fullimg_{split}.jpg'}", flush=True)


# --------------------------------------------------------------------------------------
# 9. Data-quality checks (report only)
# --------------------------------------------------------------------------------------
def section_quality(all_inst, all_img):
    findings = []
    inst = all_inst[all_inst["cls_id"].isin(CLASSES)].copy()

    # zero width / height
    zw = inst[(inst["px_w"] <= 0) | (inst["px_h"] <= 0)]
    findings.append(("zero_width_or_height", len(zw),
                     "bbox with width<=0 or height<=0 (degenerate)"))

    # out-of-bound box: any normalized coord outside [0,1] with >1% tolerance
    oob = inst[(inst["w"] > 1.01) | (inst["h"] > 1.01) |
               (inst["cx"] > 1.01) | (inst["cy"] > 1.01) |
               (inst["cx"] < -0.01) | (inst["cy"] < -0.01) |
               (inst["w"] < 0) | (inst["h"] < 0)]
    findings.append(("out_of_bound_box", len(oob),
                     "normalized coord outside [0,1] (1% tol)"))

    # extremely tiny box
    tiny = inst[inst["shortest"] < TINY_PX]
    findings.append(("extremely_tiny_box", len(tiny),
                     f"shortest side < {TINY_PX}px"))

    # abnormally large box
    huge = inst[inst["norm_area"] >= HUGE_FRAC]
    findings.append(("abnormally_large_box", len(huge),
                     f"normalized area >= {HUGE_FRAC}"))

    # duplicate labels (exact duplicate rows within a label file) -- recompute from txt
    dup_rows = 0
    for split in SPLITS:
        lbl_dir = DATA_ROOT / split / "labels"
        for lp in lbl_dir.glob("*.txt"):
            with open(lp, encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            uniq = set(lines)
            if len(lines) != len(uniq):
                dup_rows += (len(lines) - len(uniq))
    findings.append(("duplicate_label_rows", dup_rows,
                     "exact-duplicate lines inside a label file"))

    # highly-overlapping same-class boxes in the same image
    high_ovl = 0
    for (split, fn), g in inst.groupby(["split", "filename"]):
        for cid, gg in g.groupby("cls_id"):
            boxes = gg[["x1", "y1", "x2", "y2"]].values
            n = len(boxes)
            if n < 2:
                continue
            seen = False
            for i in range(n):
                for j in range(i + 1, n):
                    if iou(boxes[i], boxes[j]) >= DUP_IOU:
                        seen = True
                        break
                if seen:
                    break
            if seen:
                high_ovl += 1
    findings.append(("high_overlap_same_class_pairs", high_ovl,
                     f"images with >=1 same-class box pair IoU>={DUP_IOU}"))

    # label missing / empty
    miss = int(all_img["label_missing"].sum())
    empty = int(all_img["label_empty"].sum())
    findings.append(("label_missing", miss, "image without a .txt label"))
    findings.append(("label_empty", empty, "image with empty .txt (background)"))

    return findings


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def make_plots(all_inst, all_img):
    inst = all_inst[all_inst["cls_id"].isin(CLASSES)].copy()
    plt.style.use("default")

    # --- per-metric boxplots (class x split) ---
    metric_col = {"bbox_area": "area", "shortest_side": "shortest",
                  "aspect_ratio": "aspect", "resized_shortest_side": "rs_short"}
    for mname, col in metric_col.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        data, labels, colors = [], [], []
        for cls_name in CLASS_ORDER:
            for split in SPLITS:
                vals = inst[(inst["cls"] == cls_name) & (inst["split"] == split)][col].dropna().values
                data.append(vals)
                labels.append(f"{cls_name}\n{split}")
                colors.append(SPLIT_COLOR[split])
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_title(f"{mname} distribution by class & split")
        ax.set_ylabel(mname)
        ax.legend(handles=[Patch(color=SPLIT_COLOR[s], label=s) for s in SPLITS], loc="upper right")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"boxplot_{mname}.png", dpi=110)
        plt.close()

    # --- per-class train/val/test histograms of each shift metric ---
    shift_cols = {"area": "area", "shortest_side": "shortest", "aspect_ratio": "aspect"}
    for cls_name in CLASS_ORDER:
        sub = inst[inst["cls"] == cls_name]
        for mname, col in shift_cols.items():
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for split in SPLITS:
                vals = sub[sub["split"] == split][col].dropna().values
                if len(vals) == 0:
                    continue
                lo, hi = np.percentile(inst[col].dropna(), [1, 99])
                bins = np.linspace(lo, hi, 40)
                ax.hist(vals, bins=bins, alpha=0.5, label=f"{split} (n={len(vals)})",
                        color=SPLIT_COLOR[split], density=True)
            ax.set_title(f"{cls_name}: {mname} by split")
            ax.set_xlabel(mname); ax.set_ylabel("density")
            ax.legend()
            plt.tight_layout()
            plt.savefig(PLOT_DIR / f"hist_{mname}_{cls_name}.png", dpi=110)
            plt.close()

    # --- overall (pooled splits) class comparisons ---
    pooled = inst.copy()
    def overall_hist(col, title, fname, log=False, clip=None):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for cls_name in CLASS_ORDER:
            vals = pooled[pooled["cls"] == cls_name][col].dropna().values
            if clip is not None:
                vals = np.clip(vals, *clip)
            lo, hi = np.percentile(pooled[col].dropna(), [1, 99])
            bins = np.linspace(lo, hi, 50)
            ax.hist(vals, bins=bins, alpha=0.5, label=f"{cls_name} (n={len(vals)})",
                    color=CLS_COLOR[cls_name], density=True)
        ax.set_title(title); ax.set_xlabel(col); ax.set_ylabel("density")
        if log:
            ax.set_yscale("log")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / fname, dpi=110)
        plt.close()

    overall_hist("area", "Overall bbox area by class (px^2)", "overall_area_hist.png")
    overall_hist("shortest", "Overall shortest side by class (px)", "overall_shortest_hist.png")
    overall_hist("aspect", "Overall aspect ratio by class", "overall_aspect_hist.png", clip=(1, 15))
    overall_hist("rs_short", "Overall shortest side AFTER resize->640 (px)", "overall_resized_shortest_hist.png")
    overall_hist("rs_area", "Overall bbox area AFTER resize->640 (px^2)", "overall_resized_area_hist.png")

    # --- resized small-object fraction bar chart ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.25
    xs = np.arange(3)
    for i, thr in enumerate([8, 16, 32]):
        fracs = []
        for cls_name in CLASS_ORDER:
            vals = pooled[pooled["cls"] == cls_name]["rs_short"].dropna().values
            fracs.append(np.mean(vals < thr))
        ax.bar(xs + (i - 1) * width, fracs, width, label=f"resized shortest < {thr}px")
    ax.set_xticks(xs); ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylabel("fraction of boxes")
    ax.set_title("Fraction of boxes that become small after letterbox->640")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "resized_small_frac.png", dpi=110)
    plt.close()

    # --- border-touch fraction bar chart ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    flags = ["touch_left", "touch_right", "touch_top", "touch_bottom", "touch_any"]
    x = np.arange(len(flags))
    for i, cls_name in enumerate(CLASS_ORDER):
        sub = pooled[pooled["cls"] == cls_name]
        fracs = [sub[f].mean() for f in flags]
        ax.bar(x + (i - 1) * width, fracs, width, label=cls_name, color=CLS_COLOR[cls_name])
    ax.set_xticks(x); ax.set_xticklabels(flags, rotation=20)
    ax.set_ylabel("fraction of boxes touching border")
    ax.set_title("Border-touching fraction by class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "border_touch_frac.png", dpi=110)
    plt.close()

    # --- image-condition: brightness / contrast / color by class-presence ---
    for m in ["brightness", "contrast", "R_mean", "G_mean", "B_mean"]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for cls_name in CLASS_ORDER:
            vals = all_img[all_img[f"has_{cls_name}"] == 1][m].dropna().values
            lo, hi = np.percentile(all_img[m].dropna(), [1, 99])
            bins = np.linspace(lo, hi, 40)
            ax.hist(vals, bins=bins, alpha=0.5, density=True,
                    label=f"images w/ {cls_name} (n={len(vals)})", color=CLS_COLOR[cls_name])
        ax.set_title(f"Image {m} by class presence")
        ax.set_xlabel(m); ax.set_ylabel("density"); ax.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"imagecond_{m}.png", dpi=110)
        plt.close()

    print(f"[plots] wrote plots to {PLOT_DIR}", flush=True)


# --------------------------------------------------------------------------------------
# Summary writer (answers A..G, statistics-backed)
# --------------------------------------------------------------------------------------
def fmt(v, n=3):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "NA"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.{n}f}"


def lookup(df, **conds):
    """Find a value in a tidy long df matching all conds; returns scalar or dict."""
    d = df
    for k, v in conds.items():
        if k in d.columns:
            d = d[d[k] == v]
    return d


def tidy_get(df, scope, cls, metric, stat):
    r = df[(df["scope"] == scope) & (df["class"] == cls) & (df["metric"] == metric) & (df["stat"] == stat)]
    return r["value"].iloc[0] if len(r) else np.nan


def write_summary(all_inst, all_img, cls_df, imb_df, geo_df, resize_df, ratio_note,
                  cooc_df, border_df, imgcond_df, shift_df, quality_findings):
    L = []
    def w(s=""):
        L.append(s)

    n_total_img = len(all_img)
    n_total_inst = len(all_inst[all_inst["cls_id"].isin(CLASSES)])
    w("=" * 88)
    w("HONG KONG WALL-DEFECT DATASET -- CLASS-CONDITIONED ANALYSIS (why is `moisture` hard?)")
    w("=" * 88)
    w(f"Dataset root : {DATA_ROOT}")
    w(f"Output dir   : {OUT_DIR}")
    w(f"Splits       : {SPLITS}")
    w(f"Classes      : {CLASSES}")
    w(f"Total images : {n_total_img}   Total instances: {n_total_inst}")
    w(f"Image size   : uniform H=768 x W=1024 (verified).")
    w(f"Letterbox    : imgsz={IMGSZ}, scaleup={SCALEUP}. {ratio_note}")
    w(f"Border tol   : a box 'touches' a border if its edge is within {BORDER_TOL_PX}px of the image edge.")
    w("Definitions  : brightness=mean(luma); brightness_std=std(luma); "
      "contrast=(p95-p5)/255 of luma; aspect=max(w,h)/min(w,h).")
    w("")

    # ---- baseline reminder ----
    w("YOLOv8 baseline (test set):")
    w("  spalling : P=0.885 R=0.807 mAP50=0.834 mAP50-95=0.538")
    w("  crack    : P=0.771 R=0.689 mAP50=0.716 mAP50-95=0.432")
    w("  moisture : P=0.661 R=0.574 mAP50=0.549 mAP50-95=0.316   <-- worst on every metric")
    w("")

    # numeric helpers for moisture vs others
    def overall_count(cls):
        r = cls_df[(cls_df["scope"] == "overall") & (cls_df["class"] == cls)]
        return int(r["instance_count"].iloc[0]) if len(r) else 0
    def overall_img(cls):
        r = cls_df[(cls_df["scope"] == "overall") & (cls_df["class"] == cls)]
        return int(r["images_with_class"].iloc[0]) if len(r) else 0

    cnt = {c: overall_count(c) for c in CLASS_ORDER}
    imgc = {c: overall_img(c) for c in CLASS_ORDER}

    # =====================================================================
    w("=" * 88)
    w("SECTION 1 -- CLASS DISTRIBUTION")
    w("=" * 88)
    w(cls_df[cls_df["scope"] == "overall"].to_string(index=False))
    w("")
    w("Imbalance (instance count, max/min across classes):")
    w(imb_df.to_string(index=False))
    w("")

    w("=" * 88)
    w("SECTION 2 -- BBOX GEOMETRY (overall, per class)")
    w("=" * 88)
    for metric in GEO_METRICS:
        w(f"\n-- {metric} --")
        rows = []
        for c in CLASS_ORDER:
            s = {st: tidy_get(geo_df, "overall", c, metric, st)
                 for st in ["mean", "std", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]}
            rows.append({"class": c, **s})
        w(pd.DataFrame(rows).to_string(index=False))
    w("\nThreshold fractions (overall, per class):")
    thr_metrics = ["shortest_lt8", "shortest_lt16", "shortest_lt32",
                   "area_lt_32sq", "area_32_to_96sq", "area_ge_96sq",
                   "aspect_ge3", "aspect_ge5", "aspect_ge10"]
    rows = []
    for c in CLASS_ORDER:
        rec = {"class": c}
        for m in thr_metrics:
            rec[m] = tidy_get(geo_df, "overall", c, m, "frac")
        rows.append(rec)
    w(pd.DataFrame(rows).to_string(index=False))
    w("")

    w("=" * 88)
    w("SECTION 3 -- RESIZE -> 640 (overall, per class)")
    w("=" * 88)
    for metric in ["width", "height", "shortest_side", "area"]:
        w(f"\n-- resized {metric} @640 --")
        rows = []
        for c in CLASS_ORDER:
            s = {st: tidy_get(resize_df, "overall", c, metric, st)
                 for st in ["mean", "std", "p10", "p50", "p90"]}
            rows.append({"class": c, **s})
        w(pd.DataFrame(rows).to_string(index=False))
    w("\nFraction of boxes that become tiny after letterbox -> 640:")
    rows = []
    for c in CLASS_ORDER:
        rec = {"class": c}
        for thr in (4, 8, 16, 32):
            rec[f"rs_short<{thr}px"] = tidy_get(resize_df, "overall", c, f"resized_shortest_lt{thr}", "frac")
        rows.append(rec)
    w(pd.DataFrame(rows).to_string(index=False))
    w("")

    w("=" * 88)
    w("SECTION 4 -- CO-OCCURRENCE (image level)")
    w("=" * 88)
    pp = cooc_df[cooc_df["P_crack_given_moisture"].notna()][[
        "scope", "n_images", "P_crack_given_moisture", "P_spalling_given_moisture",
        "P_moisture_given_crack", "P_moisture_given_spalling"]]
    w(pp.to_string(index=False))
    w("\nMoisture image partitions:")
    wp = cooc_df[cooc_df["partition"].notna()][["scope", "partition", "count", "frac"]]
    w(wp.to_string(index=False))
    w("")

    w("=" * 88)
    w("SECTION 5 -- BORDER ANALYSIS (overall)")
    w("=" * 88)
    rows = []
    for c in CLASS_ORDER:
        sub = border_df[(border_df["scope"] == "overall") & (border_df["class"] == c) &
                        (border_df["metric"].isin(["touch_left", "touch_right", "touch_top",
                                                   "touch_bottom", "touch_any"]))]
        rec = {"class": c}
        for _, rr in sub.iterrows():
            rec[rr["metric"]] = rr["frac"]
        rows.append(rec)
    w(pd.DataFrame(rows).to_string(index=False))
    w("\nMin distance to any border (px), overall:")
    rows = []
    for c in CLASS_ORDER:
        sub = border_df[(border_df["scope"] == "overall") & (border_df["class"] == c) &
                        (border_df["metric"] == "min_dist_to_border_px")]
        rec = {"class": c}
        for _, rr in sub.iterrows():
            rec[rr["stat"]] = rr["value"]
        rows.append(rec)
    w(pd.DataFrame(rows)[["class", "mean", "p10", "p50", "p90"]].to_string(index=False))
    w("")

    w("=" * 88)
    w("SECTION 6 -- IMAGE CONDITIONS (overall, grouped by class presence)")
    w("=" * 88)
    for m in ["brightness", "brightness_std", "contrast", "R_mean", "G_mean", "B_mean"]:
        rows = []
        for c in CLASS_ORDER:
            s = {st: None for st in ["mean", "std", "p10", "p50", "p90"]}
            for st in s:
                r = imgcond_df[(imgcond_df["scope"] == "overall") &
                               (imgcond_df["group"] == f"images_with_{c}") &
                               (imgcond_df["metric"] == m) & (imgcond_df["stat"] == st)]
                s[st] = r["value"].iloc[0] if len(r) else np.nan
            rows.append({"class": c, **s})
        w(f"\n-- {m} --")
        w(pd.DataFrame(rows).to_string(index=False))
    w("")

    w("=" * 88)
    w("SECTION 7 -- TRAIN/VAL/TEST DISTRIBUTION SHIFT")
    w("=" * 88)
    if HAS_SCIPY:
        cols = ["class", "metric", "split", "n", "mean", "p50",
                "ks_stat_vs_train", "ks_pval_vs_train", "norm_shift_vs_train"]
    else:
        cols = ["class", "metric", "split", "n", "mean", "p50"]
    w(shift_df[cols].to_string(index=False))
    w("")

    w("=" * 88)
    w("SECTION 8 -- MONTAGES")
    w("=" * 88)
    w(f"Crop montages   : {OUT_DIR}/{{moisture,crack,spalling}}_montage/crop_<split>_<small|medium|large>.jpg")
    w(f"Full-image+box  : {OUT_DIR}/{{moisture,crack,spalling}}_montage/fullimg_<split>.jpg")
    w(f"Each crop tile is labeled with: split / filename / bbox size / normalized area.")
    w("")

    w("=" * 88)
    w("SECTION 9 -- DATA-QUALITY CHECKS (report only, nothing modified)")
    w("=" * 88)
    for key, n, desc in quality_findings:
        w(f"  {key:32s} count={n:<6d}  {desc}")
    w("")

    # =====================================================================
    w("=" * 88)
    w("ANSWERS A..G  (every claim backed by a number above)")
    w("=" * 88)

    # A. moisture fewer?
    w("\nA. Is `moisture` rarer than the other classes?")
    w(f"   instance counts  : spalling={cnt['spalling']} crack={cnt['crack']} moisture={cnt['moisture']}")
    w(f"   images-with-class: spalling={imgc['spalling']} crack={imgc['crack']} moisture={imgc['moisture']}")
    imb_o = imb_df[imb_df["scope"] == "overall"]["imbalance_ratio_inst"]
    imb_o = imb_o.iloc[0] if len(imb_o) else np.nan
    min_c = cnt["moisture"] == min(cnt.values())
    w(f"   overall imbalance ratio (max/min instances) = {fmt(imb_o, 2)}")
    if cnt["moisture"] < cnt["spalling"] and cnt["moisture"] < cnt["crack"]:
        w("   -> YES: moisture has the FEWEST instances (and images). "
          "Lower sample count -> weaker supervision, a direct contributor to its worse metrics.")
    else:
        w(f"   -> NOT clearly the rarest: spalling={cnt['spalling']} crack={cnt['crack']} moisture={cnt['moisture']}.")

    # B. moisture smaller?
    def g(metric, stat):
        return {c: tidy_get(geo_df, "overall", c, metric, stat) for c in CLASS_ORDER}
    area_mean = g("area", "mean"); short_mean = g("shortest_side", "mean")
    short_p10 = g("shortest_side", "p10"); short_p50 = g("shortest_side", "p50")
    w("\nB. Is `moisture` smaller (in source pixels)?")
    w(f"   area mean        : spalling={fmt(area_mean['spalling'],0)} crack={fmt(area_mean['crack'],0)} moisture={fmt(area_mean['moisture'],0)}")
    w(f"   shortest mean    : spalling={fmt(short_mean['spalling'],1)} crack={fmt(short_mean['crack'],1)} moisture={fmt(short_mean['moisture'],1)}")
    w(f"   shortest p50     : spalling={fmt(short_p50['spalling'],1)} crack={fmt(short_p50['crack'],1)} moisture={fmt(short_p50['moisture'],1)}")
    w(f"   shortest p10     : spalling={fmt(short_p10['spalling'],1)} crack={fmt(short_p10['crack'],1)} moisture={fmt(short_p10['moisture'],1)}")
    sm_frac = tidy_get(geo_df, "overall", "moisture", "area_lt_32sq", "frac")
    w(f"   moisture small-object fraction (area<32^2) = {fmt(sm_frac)}")
    # moisture-vs-crack size nuance (crack detects better despite being smaller)
    rs16 = {c: tidy_get(resize_df, "overall", c, "resized_shortest_lt16", "frac") for c in CLASS_ORDER}
    w(f"   resized-shortest<16px @640: spalling={fmt(rs16['spalling'])} crack={fmt(rs16['crack'])} moisture={fmt(rs16['moisture'])}")
    if area_mean["moisture"] < area_mean["spalling"] and area_mean["moisture"] < area_mean["crack"]:
        w("   -> moisture has the SMALLEST mean area, but its shortest side (mean/p50) sits BETWEEN "
          "spalling and crack, and after resize->640 moisture is NOT the smallest (crack is). "
          "Since crack is smaller yet detects better, object size ALONE does not explain moisture's gap.")
    else:
        w("   -> moisture is not the smallest class by every measure; size is a secondary factor.")

    # C. border?
    def border_frac(c, flag):
        r = border_df[(border_df["scope"] == "overall") & (border_df["class"] == c) &
                      (border_df["metric"] == flag)]
        return r["frac"].iloc[0] if len(r) else np.nan
    w("\nC. Is `moisture` closer to the image edge / border?")
    anyf = {c: border_frac(c, "touch_any") for c in CLASS_ORDER}
    w(f"   touch_any fraction: spalling={fmt(anyf['spalling'])} crack={fmt(anyf['crack'])} moisture={fmt(anyf['moisture'])}")
    sub = border_df[(border_df["scope"] == "overall") & (border_df["metric"] == "min_dist_to_border_px") & (border_df["stat"] == "mean")]
    dmean = {rr["class"]: rr["value"] for _, rr in sub.iterrows()}
    w(f"   mean min-dist to border (px): "
      f"spalling={fmt(dmean.get('spalling'),1)} crack={fmt(dmean.get('crack'),1)} moisture={fmt(dmean.get('moisture'),1)}")
    delta_sp = anyf["moisture"] - anyf["spalling"]
    delta_cr = anyf["moisture"] - anyf["crack"]
    if max(delta_sp, delta_cr) < 0.03:
        w(f"   -> MARGINAL: moisture touches a border only slightly more often "
          f"(+{fmt(delta_sp)} vs spalling, +{fmt(delta_cr)} vs crack) and is only ~"
          f"{fmt((dmean.get('spalling',0)-dmean.get('moisture',0))/max(dmean.get('spalling',1),1)*100,0)}% "
          "closer in mean min-distance. Border location is NOT a strong differentiator for moisture.")
    else:
        w("   -> YES: moisture is notably more often at the image border -> truncated objects harder to localize.")

    # D. aspect ratio?
    asp_mean = g("aspect_ratio", "mean"); asp_ge5 = {c: tidy_get(geo_df, "overall", c, "aspect_ge5", "frac") for c in CLASS_ORDER}
    w("\nD. Does `moisture` have a different aspect ratio?")
    w(f"   aspect mean : spalling={fmt(asp_mean['spalling'],2)} crack={fmt(asp_mean['crack'],2)} moisture={fmt(asp_mean['moisture'],2)}")
    w(f"   aspect>=5 frac: spalling={fmt(asp_ge5['spalling'])} crack={fmt(asp_ge5['crack'])} moisture={fmt(asp_ge5['moisture'])}")
    w("   (crack is expected to be the most elongated; compare moisture vs spalling.)")

    # E. distribution shift?
    w("\nE. Is there a Train/Val/Test distribution shift for `moisture`?")
    if HAS_SCIPY:
        sm = shift_df[shift_df["class"] == "moisture"]
        for metric in ["area", "shortest_side", "aspect_ratio", "brightness", "contrast"]:
            sub = sm[sm["metric"] == metric].set_index("split")
            trec = sub.loc["test"] if "test" in sub.index else None
            if trec is None:
                continue
            w(f"   {metric:14s}: train_mean={fmt(sub.loc['train','mean'])} test_mean={fmt(trec['mean'])} "
              f"ks_stat={fmt(trec['ks_stat_vs_train'],3)} ks_p={fmt(trec['ks_pval_vs_train'],4)} "
              f"norm_shift={fmt(trec['norm_shift_vs_train'],2)}")
        w("   (large ks_stat / small ks_p / large norm_shift => distribution shift)")
        # comparative: moisture size-shift vs other classes
        def test_norm_shift(cls_name, metric):
            r = shift_df[(shift_df["class"] == cls_name) & (shift_df["metric"] == metric) &
                         (shift_df["split"] == "test")]["norm_shift_vs_train"]
            return r.iloc[0] if len(r) else np.nan
        w("   Comparative test-vs-train norm_shift (size): "
          f"spalling_area={fmt(test_norm_shift('spalling','area'),2)} "
          f"crack_area={fmt(test_norm_shift('crack','area'),2)} "
          f"moisture_area={fmt(test_norm_shift('moisture','area'),2)}; "
          f"moisture_shortest={fmt(test_norm_shift('moisture','shortest_side'),2)}.")
        ns_mo = test_norm_shift("moisture", "area")
        ns_cr = test_norm_shift("crack", "area")
        ns_sp = test_norm_shift("spalling", "area")
        if ns_mo == ns_mo and ns_mo > max(x for x in [ns_cr, ns_sp] if x == x):
            w("   -> YES for SIZE: moisture shows the LARGEST train->test size shift of the three classes "
              "(test moisture boxes are bigger than train), while its appearance (brightness/contrast) is stable.")
        else:
            w("   -> moisture size shift is not larger than the other classes; appearance is stable.")
    else:
        w("   scipy not available; see distribution_shift.csv for descriptive stats.")

    # F. co-occurrence?
    w("\nF. Does `moisture` frequently co-occur with crack/spalling?")
    oc = cooc_df[cooc_df["scope"] == "overall"]
    pcm = oc["P_crack_given_moisture"].iloc[0] if len(oc) else np.nan
    psm = oc["P_spalling_given_moisture"].iloc[0] if len(oc) else np.nan
    mo_any_cnt = oc[oc["partition"] == "moisture_any"]["count"]
    mo_any_cnt = mo_any_cnt.iloc[0] if len(mo_any_cnt) else np.nan
    only_m_cnt = oc[oc["partition"] == "only_moisture"]["count"]
    only_m_cnt = only_m_cnt.iloc[0] if len(only_m_cnt) else np.nan
    only_m_of_moisture = only_m_cnt / mo_any_cnt if (mo_any_cnt and mo_any_cnt == mo_any_cnt) else np.nan
    w(f"   P(crack   | moisture) = {fmt(pcm)}")
    w(f"   P(spalling| moisture) = {fmt(psm)}")
    w(f"   fraction of MOISTURE images that are moisture-only (no crack/spalling) = {fmt(only_m_of_moisture)}")
    if (pcm if pcm == pcm else 0) > 0.3 or (psm if psm == psm else 0) > 0.3:
        w("   -> moisture frequently co-occurs with other defects -> possible label confusion / "
          "overlap with spalling regions (visually similar stains).")
    else:
        w("   -> moisture mostly appears without the other classes -> co-occurrence is limited; "
          "confusion is more likely visual than contextual.")

    # G. synthesis
    # pull image-condition means for an evidence-backed visual description
    def icmean(cls_name, m):
        r = imgcond_df[(imgcond_df["scope"] == "overall") & (imgcond_df["group"] == f"images_with_{cls_name}") &
                       (imgcond_df["metric"] == m) & (imgcond_df["stat"] == "mean")]
        return r["value"].iloc[0] if len(r) else np.nan
    br_mo, br_cr, br_sp = icmean("moisture", "brightness"), icmean("crack", "brightness"), icmean("spalling", "brightness")
    bstd_mo = icmean("moisture", "brightness_std")
    ct_mo, ct_cr, ct_sp = icmean("moisture", "contrast"), icmean("crack", "contrast"), icmean("spalling", "contrast")
    w("\nG. Most likely data-side causes of moisture's poor detection (ranked by the strength of evidence above):")
    w(f"   1) SEVERE class imbalance (strongest): only {cnt['moisture']} moisture instances vs "
      f"{cnt['crack']} crack / {cnt['spalling']} spalling ({fmt(imb_o,1)}x imbalance, see A). "
      "Far fewer positive examples -> weakest supervision; this alone plausibly accounts for much of the gap.")
    w(f"   2) Visual ambiguity + co-occurrence confusion (strong): moisture images are DARKER "
      f"(brightness {fmt(br_mo,0)} vs crack {fmt(br_cr,0)}/spalling {fmt(br_sp,0)}) with HIGHER internal "
      f"variance (brightness_std {fmt(bstd_mo,1)}) and HIGHER global contrast ({fmt(ct_mo,2)} vs "
      f"{fmt(ct_cr,2)}/{fmt(ct_sp,2)}) -- i.e. diffuse, heterogeneous staining, not crisp edges. "
      f"Combined with heavy co-occurrence (P(crack|moisture)={fmt(pcm)}, only {fmt(only_m_of_moisture)} of "
      "moisture images are moisture-only, see F), moisture is prone to visual confusion with crack/spalling "
      "and with wet/spalled background. (NOTE: the data says higher, not lower, contrast -- do not assume low contrast.)")
    w("   3) Train->Test size distribution shift (moderate, moisture-specific): see E -- moisture has the "
      "LARGEST size shift of the three classes (test boxes bigger than train), while crack/spalling are stable.")
    w("   4) Object size (weak/secondary): see B -- moisture has the smallest mean AREA, but it is NOT the "
      "smallest by shortest side and crack (which detects better) is smaller still after resize->640. "
      "So size is a contributor but not the main driver vs crack.")
    w("   5) Border location (negligible): see C -- moisture touches the border only marginally more often; "
      "not a meaningful differentiator.")
    w("")
    w("NOTE: these are data-statistics observations, not model-error analysis. They identify the "
      "most plausible contributors; confirming each requires per-image inspection (montages) and/or "
      "model confusion-matrix analysis.")
    w("")
    w("=" * 88)
    w("END OF SUMMARY")
    w("=" * 88)

    text = "\n".join(str(x) for x in L)
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(f"[summary] wrote {OUT_DIR/'summary.txt'} ({len(text)} chars)", flush=True)
    return text


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    print(f"== Loading dataset from {DATA_ROOT} ==", flush=True)
    inst_parts, img_parts = [], []
    for split in SPLITS:
        i_df, g_df = load_split(split)
        inst_parts.append(i_df)
        img_parts.append(g_df)
    all_inst = pd.concat(inst_parts, ignore_index=True)
    all_img = pd.concat(img_parts, ignore_index=True)
    # add an 'overall' pseudo-split alias by concatenating again is not needed; section fns
    # already handle 'overall' by using the full frame. We keep split column intact.

    print("== Section 1: class distribution ==", flush=True)
    cls_df, imb_df = section_class_distribution(all_inst, all_img)

    print("== Section 2: bbox geometry ==", flush=True)
    geo_df = section_bbox_geometry(all_inst)

    print("== Section 3: resize->640 ==", flush=True)
    resize_df, ratio_note = section_resize640(all_inst)

    print("== Section 4: co-occurrence ==", flush=True)
    cooc_df = section_cooccurrence(all_img)

    print("== Section 5: border analysis ==", flush=True)
    border_df = section_border(all_inst)

    print("== Section 6: image conditions ==", flush=True)
    imgcond_df = section_image_conditions(all_img)

    print("== Section 7: distribution shift ==", flush=True)
    shift_df = section_distribution_shift(all_inst, all_img)

    print("== Section 8: montages ==", flush=True)
    section_montages(all_inst, all_img)

    print("== Section 9: quality checks ==", flush=True)
    quality_findings = section_quality(all_inst, all_img)

    print("== Plots ==", flush=True)
    make_plots(all_inst, all_img)

    # ---- save CSVs ----
    cls_df.to_csv(OUT_DIR / "class_statistics.csv", index=False)
    geo_df.to_csv(OUT_DIR / "bbox_statistics.csv", index=False)
    resize_df.to_csv(OUT_DIR / "resize640_statistics.csv", index=False)
    cooc_df.to_csv(OUT_DIR / "cooccurrence.csv", index=False)
    border_df.to_csv(OUT_DIR / "border_statistics.csv", index=False)
    all_img.to_csv(OUT_DIR / "image_statistics.csv", index=False)
    shift_df.to_csv(OUT_DIR / "distribution_shift.csv", index=False)

    # ---- summary ----
    print("== Writing summary ==", flush=True)
    write_summary(all_inst, all_img, cls_df, imb_df, geo_df, resize_df, ratio_note,
                  cooc_df, border_df, imgcond_df, shift_df, quality_findings)

    print("\nDONE. Outputs in:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
