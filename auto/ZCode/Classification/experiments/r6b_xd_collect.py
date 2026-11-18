#!/usr/bin/env python3
"""R6-B 跨域汇总: BUSI/covid/CUB 上 6 模型(convnext_tiny 基线, CLE-Net, Jumbo-small, GmNet-s3, iFormer-S,
LSNet-S) 3 seeds 的 Macro-F1/κ/AUROC/恶性或各类敏感度均值±样本std, 同 seed 配对 Δ(CLE-Net − 该网络);
另附同环境参照行 mobilenet_v3_large(来自 r6 批 weights_xd_*_r6)。
用法: python experiments/r6b_xd_collect.py [--json experiments/r6b_xd_summary.json]
"""
import argparse
import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = (42, 43, 44)
DOMAINS = [("busi", "BUSI"), ("covid", "covid"), ("cub", "CUB-200")]
ORDER = ["convnext_tiny", "convnext_egb_scg", "jumbo_small", "gmnet_s3", "iformer_s", "lsnet_s"]
NICE = {"convnext_tiny": "ConvNeXt-T 基线", "convnext_egb_scg": "CLE-Net (egb_scg)",
        "jumbo_small": "Jumbo-small (ICLR'26)", "gmnet_s3": "GmNet-s3 (ICLR'26)",
        "iformer_s": "iFormer-S (ICLR'25)", "lsnet_s": "LSNet-S (CVPR'25)",
        "mobilenet_v3_large": "MobileNetV3-L (参照)"}


def runs(root, model):
    out = {}
    for f in glob.glob(os.path.join(REPO, root, f"{model}_seed*", model, "eval_*", "results.json")):
        r = json.load(open(f))
        if os.path.basename(r.get("test_dir", "")).rstrip("/") == "test":
            s = int(f.split("_seed")[1].split("/")[0])
            if s not in out or f > out[s][0]:
                out[s] = (f, r)
    return {s: v[1] for s, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(REPO, "experiments", "r6b_xd_summary.json"))
    a = ap.parse_args()
    res = {}
    for dom, nice in DOMAINS:
        root = f"weights_xd_{dom}_r6b"
        if not os.path.isdir(os.path.join(REPO, root)):
            print(f"{nice}: 批次目录不存在, 跳过")
            continue
        print(f"\n===== {nice} (3 seeds 42/43/44, 同批同协议) =====")
        print(f"{'方法':26s} {'Macro-F1':>12s} {'Kappa':>12s} {'AUROC':>12s}   F1 逐 seed          CLE-Net−其")
        base = runs(root, "convnext_egb_scg")
        dtab = {}
        rows = []
        for m in ORDER + ["mobilenet_v3_large"]:
            r = runs(root, m)
            if not r:
                continue
            seeds = sorted(set(r) & set(S))
            f1 = np.array([r[s]["macro_f1"] * 100 for s in seeds])
            kap = np.array([r[s]["kappa"] * 100 for s in seeds])
            auc = np.array([r[s]["auc_roc_macro"] * 100 for s in seeds])
            d = None
            if m != "convnext_egb_scg":
                pair = [s for s in seeds if s in base]
                d = np.array([base[s]["macro_f1"] * 100 - r[s]["macro_f1"] * 100 for s in pair]) if pair else None
            row = {"f1_mean": round(float(f1.mean()), 2), "f1_std": round(float(f1.std(ddof=1)), 2),
                   "kappa": round(float(kap.mean()), 2), "auroc": round(float(auc.mean()), 2),
                   "per_seed_f1": [round(x, 2) for x in f1],
                   "dF1_vs_clenet": None if d is None else round(float(d.mean()), 2),
                   "win_tie_loss": None if d is None else f"{int((d > 0).sum())}/{int((d == 0).sum())}/{int((d < 0).sum())}",
                   "n": len(seeds)}
            res.setdefault(dom, {})[NICE[m]] = row
            dstr = "—" if d is None else f"{d.mean():+.2f} ({row['win_tie_loss']})"
            print(f"{NICE[m]:26s} {f1.mean():6.2f}±{f1.std(ddof=1):4.2f} {kap.mean():8.2f} {auc.mean():8.2f}   "
                  f"{'/'.join(f'{x:.2f}' for x in f1):16s}  {dstr}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, ensure_ascii=False)
        print("\nsaved ->", a.json)


if __name__ == "__main__":
    main()
