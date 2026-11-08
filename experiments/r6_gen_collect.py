#!/usr/bin/env python3
"""R6 泛化汇总: 各域 5 配置链(baseline / ①fz16 / ①+②fz16_dce16 / ①+②+③fz16_dce4m[最终] / ②+③dce4m[无①])
的 Macro-F1 均值±std 与同 seed 配对 Δ(最终 vs baseline)。
用法: python experiments/r6_gen_collect.py [busi heel covid cub tcga]
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["mobilenet_v3_large", "mbv3_r6_fz16", "mbv3_fz16_dce16", "mbv3_fz16_dce4m", "mbv3_dce4m"]
NICE = {"mobilenet_v3_large": "baseline", "mbv3_r6_fz16": "①fz16", "mbv3_fz16_dce16": "①+②dce16",
        "mbv3_fz16_dce4m": "①+②+③dce4m(final)", "mbv3_dce4m": "②+③dce4m(no-①)"}


def load(root, model):
    import glob
    out = {}
    for f in glob.glob(os.path.join(root, f"{model}_seed*", model, "eval_*", "results.json")):
        r = json.load(open(f))
        if os.path.basename(r.get("test_dir", "")) == "test":
            out[int(f.split("_seed")[1].split("/")[0])] = r["macro_f1"] * 100
    return out


def main():
    doms = sys.argv[1:] or ["busi", "heel", "covid", "cub", "tcga"]
    print(f"{'domain':8s} {'config':22s} n  F1(mean±std)   Δfinal-baseline(per-seed)" + " " * 20)
    for dom in doms:
        root = os.path.join(REPO, f"weights_xd_{dom}_r6")
        if not os.path.isdir(root):
            print(f"{dom:8s} (pending)")
            continue
        runs = {m: load(root, m) for m in ORDER}
        base = runs["mobilenet_v3_large"]
        fin = runs["mbv3_fz16_dce4m"]
        pair = sorted(set(base) & set(fin))
        for m in ORDER:
            r = runs[m]
            if not r:
                continue
            seeds = sorted(r)
            d = f"Δ {np.mean([fin[s] - base[s] for s in pair]):+.2f} ({int(sum(fin[s] > base[s] for s in pair))}/{len(pair)})"
            print(f"{dom:8s} {NICE[m]:22s} {len(seeds)}  {np.mean(list(r.values())):5.2f}±{np.std(list(r.values())):4.2f}   {d if m=='mbv3_fz16_dce4m' else ''}")
        print(f"{dom:8s} per-seed final {[round(fin[s],2) for s in pair]} vs base {[round(base[s],2) for s in pair]}")
    print("done")


if __name__ == "__main__":
    main()
