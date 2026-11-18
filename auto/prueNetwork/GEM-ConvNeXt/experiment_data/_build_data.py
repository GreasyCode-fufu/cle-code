#!/usr/bin/env python3
"""GEM-ConvNeXt 交付包数据整理: 从 ZCode/Classification 的原始记录生成 experiment_data/*.csv"""
import csv
import glob
import json
import os

import numpy as np

SRC = "/home/login/data/cle/auto/ZCode/Classification"
DST = "/home/login/data/cle/auto/prueNetwork/GEM-ConvNeXt"
KEYS = ["macro_f1", "accuracy", "weighted_f1", "balanced_accuracy", "kappa", "mcc",
        "macro_sensitivity", "macro_specificity", "auc_roc_macro"]
ORDER = ["convnext_tiny", "convnext_v2m", "convnext_v2c", "convnext_v6f"]
LABEL = {"convnext_tiny": "baseline", "convnext_v2m": "+MSP",
         "convnext_v2c": "+MSP+DCH", "convnext_v6f": "+MSP+DCH+FGS (final)"}


def test_json(root, model, seed):
    for f in sorted(glob.glob(f"{SRC}/{root}/{model}_seed{seed}/{model}/eval_*/results.json")):
        r = json.load(open(f))
        if r.get("test_dir", "").rstrip("/").endswith("/test"):
            return r
    return None


# ── 1) CLE 主结果(逐 seed 9 指标 + MEAN) ─────────────────────────────
rows = []
for model in ORDER:
    per = []
    for s in (42, 43, 44):
        r = test_json("weights_cle_ac_v1_c", model, s)
        row = {"model": LABEL[model], "seed": s}
        row.update({k: round(r[k], 4) for k in KEYS})
        per.append(row)
        rows.append(row)
    m = {"model": LABEL[model], "seed": "MEAN±STD"}
    m.update({k: f"{np.mean([p[k] for p in per]):.4f}±{np.std([p[k] for p in per]):.4f}" for k in KEYS})
    rows.append(m)
with open(f"{DST}/experiment_data/cle_main_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "seed"] + KEYS)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# ── 2) 跨域消融(5 域 × 4 配置, F1; BUSI 仅 baseline/最终) ────────────
DOMS = {  # domain -> {model: [(results_csv, weights_root)]}
    "heelBone": {"convnext_tiny": [("xd_heel_r3", "weights_xd_heel_r3")],
                 "convnext_v2m": [("xd_heel_abl", "weights_xd_heel_abl")],
                 "convnext_v2c": [("xd_heel_r4", "weights_xd_heel_r4")],
                 "convnext_v6f": [("xd_heel_r5", "weights_xd_heel_r5")]},
    "covid_v1": {"convnext_tiny": [("xd_covid_r3", "weights_xd_covid_r3")],
                 "convnext_v2m": [("xd_covid_abl", "weights_xd_covid_abl")],
                 "convnext_v2c": [("xd_covid_r4", "weights_xd_covid_r4")],
                 "convnext_v6f": [("xd_covid_r5", "weights_xd_covid_r5")]},
    "CUB_200": {"convnext_tiny": [("xd_cub_r3", "weights_xd_cub_r3")],
                "convnext_v2m": [("xd_cub_abl", "weights_xd_cub_abl")],
                "convnext_v2c": [("xd_cub_abl", "weights_xd_cub_abl"), ("xd_cub_r4", "weights_xd_cub_r4")],
                "convnext_v6f": [("xd_cub_r5", "weights_xd_cub_r5")]},
    "TCGA_6c": {"convnext_tiny": [("xd_tcga_r5", "weights_xd_tcga_r5")],
                "convnext_v2m": [("xd_tcga_abl", "weights_xd_tcga_abl")],
                "convnext_v2c": [("xd_tcga_abl", "weights_xd_tcga_abl")],
                "convnext_v6f": [("xd_tcga_r5", "weights_xd_tcga_r5")]},
    "BUSI": {"convnext_tiny": [("xd_busi_r3", "weights_xd_busi_r3")],
             "convnext_v6f": [("xd_busi_r5", "weights_xd_busi_r5")]},
}
rows = []
for dom, mm in DOMS.items():
    for model in ORDER:
        if model not in mm:
            continue
        vals = []
        per = {}
        for csvname, root in mm[model]:
            with open(f"{SRC}/experiments/results_{csvname}.csv") as f:
                for r in csv.DictReader(f):
                    if r["model"] == model and r["status"] == "ok" and r["test_f1"]:
                        per[int(r["seed"])] = float(r["test_f1"])
        vals = [per.get(s) for s in (42, 43, 44)]
        if all(v is None for v in vals):
            continue
        rows.append({"dataset": dom, "model": LABEL[model],
                     "seed42": vals[0], "seed43": vals[1], "seed44": vals[2],
                     "mean": round(np.mean([v for v in vals if v is not None]), 4),
                     "n": sum(v is not None for v in vals)})
with open(f"{DST}/experiment_data/crossdomain_ablation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "model", "seed42", "seed43", "seed44", "mean", "n"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

# ── 3) 负结果筛选汇总(R3-R5 全部候选 vs v2c 同批) ────────────────────
CAND = {  # name -> (batch_prefix filter off, 2-seed vs v2c 96.315 delta)
    "WR@all_stages(v3f)": -0.13, "SOP_head(v3s)": -0.86, "XRM(v3x)": -0.19, "DRC(v3r)": -0.55,
    "v2c+XRM_gentle(v5xrm2)": +0.09, "v2c+SE(v5se)": -0.32, "v2c+GRN@34(v5grn)": -0.51,
    "v2c+TeLU(v5telu)": -0.84, "v2c+GRN@4(v5grn4)": -0.15, "v2c+GRN4+XRM(v5grnx)": -0.60,
    "v2c+XRM+cw1.3(v5xrcw)": -0.76, "v2c+EMA0.999": -0.40, "v2c+ls0.05": -1.29,
    "v2c+cw1,1.5,1": -0.06, "v2c+focal2": -2.40, "v2c+ACM_m0.1": -0.69, "v2c+ACM_m0.2": -0.03,
    "v2c+soup3(seeds)": -0.31, "v2c+PFESA(v5pf)": -0.78, "v2c+SDGW(v5sd)": -0.49,
    "v2c+CSAP(v6c)": -0.32, "v2c+CSAP_soft(v6c2)": -0.32, "v2c+FGS(v6f)_F1": -0.06,
    "CSM_full@a(v4a)": -2.73, "CSM@4b(v4b)": -1.63, "CSM@34c_3.8M(v4c)": -2.17,
}
with open(f"{DST}/experiment_data/negative_screening_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["candidate_on_v2c_base", "delta_F1_vs_v2c_2seed_pt", "verdict"])
    for k, v in CAND.items():
        w.writerow([k, round(v, 2), "kept(FGS, clinical gain)" if k.startswith("v2c+FGS") else
                    ("borderline" if v > -0.10 else "rejected")])

# ── 4) 复杂度 ─────────────────────────────────────────────────────────
comp = {"convnext_tiny": (27.82, 4.45, 848), "convnext_v2m": (28.27, 4.63, 842),
        "convnext_v2c": (28.27, 4.63, 835), "convnext_v6f": (28.27, 4.63, 820)}
with open(f"{DST}/experiment_data/model_complexity.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "params_M", "macs_G_thop", "fps_b1_idle"])
    w.writerow(["# 口径: thop MACs@224; FPS=RTX4090独占 batch1 200次均值(2026-11-08); params含分类头(nc=3)"])
    for m, (p, g, s) in comp.items():
        w.writerow([LABEL[m] + f" ({m})", p, g, s])

print("experiment_data 生成完成:")
for f in sorted(glob.glob(f"{DST}/experiment_data/*")):
    print(" ", os.path.basename(f))
