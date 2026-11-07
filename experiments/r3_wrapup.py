#!/usr/bin/env python3
"""R3 收尾: 独占 GPU 的 FPS 复测 + 各域全指标汇总. 由 chain_r3_wrapup.sh 在泛化链结束后调用."""
import json
import os
import subprocess
import time

import torch
from thop import profile

from cle_fair_common import build_model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "experiments", "R3_WRAPUP.md")


def bench(name, nc=3):
    m = build_model(name, num_classes=nc).cuda().eval()
    x = torch.randn(1, 3, 224, 224).cuda()
    macs, _ = profile(m, inputs=(x,), verbose=False)
    with torch.no_grad():
        for _ in range(30):
            m(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(200):
            m(x)
        torch.cuda.synchronize()
    fps = 200 / (time.time() - t0)
    n = sum(p.numel() for p in m.parameters())
    del m
    torch.cuda.empty_cache()
    return n / 1e6, macs / 1e9, fps


def collect(model, seeds, weights_root):
    keys = ["macro_f1", "kappa", "mcc", "balanced_accuracy", "auc_roc_macro"]
    rows = []
    for s in seeds:
        import glob
        fs = sorted(glob.glob(os.path.join(REPO, weights_root, f"{model}_seed{s}", model,
                                           "eval_*", "results.json")))
        r = None
        for f in fs:
            d = json.load(open(f))
            if "/test" in d.get("test_dir", "") or d.get("test_dir", "").endswith("test"):
                r = d
        if r:
            rows.append({k: r.get(k, float("nan")) * 100 for k in keys})
    if not rows:
        return None
    import numpy as np
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])),
                [round(r[k], 2) for r in rows]) for k in keys}


def main():
    lines = ["# R3 收尾汇总 (自动生成 %s)\n" % time.strftime("%F %T")]
    lines.append("## 独占 GPU 复杂度/FPS(batch=1, 224²)\n")
    lines.append("| model | params(M) | thop MACs(G) | FPS(b1) |")
    lines.append("|---|---:|---:|---:|")
    for nm in ["convnext_tiny", "convnext_v2c", "convnext_v4e", "convnext_v4c"]:
        p, g, f = bench(nm)
        lines.append(f"| {nm} | {p:.2f} | {g:.2f} | {f:.1f} |")
        print(nm, p, g, f, flush=True)
    lines.append("\n注: thop 不计 rfft/irfft 算子, v4e/v4c 实际算力高于表中 MACs。\n")

    lines.append("\n## TCGA 全指标(v4e vs tiny, 3 seeds)\n")
    for m in ["convnext_v4e", "convnext_tiny"]:
        c = collect(m, [42, 43, 44], "weights_xd_tcga_r3")
        if c:
            lines.append(f"- **{m}**: " + "  ".join(
                f"{k}={v[0]:.2f}±{v[1]:.2f}" for k, v in c.items()))
        else:
            lines.append(f"- {m}: 未找到结果")
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
