#!/usr/bin/env python3
"""R6 P0b — /16 证据池化方式探针 (2026-11-08)

问题: 星乘头的 /16 证据用 GAP(f12)。恶性改变可能是局灶性的, 且 25–31% 视野外暗背景稀释 GAP。
比较 f12 的池化: GAP / FOV 掩码 GAP(只在组织区平均) / GeM(p=3) / max / FOV 掩码 GeM,
在冻结特征(ImageNet 预训练 + 10-26 历史微调 s42-44)上做线性探针(与 r6_p0_analysis 同协议),
报告 test Macro-F1 / sens_M / B-vs-M AUC, 以及与 f16 GAP 拼接后的增量。
输出: experiments/r6_p0b_pooling.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
from r6_p0_analysis import (list_split, load_tensors, probe, PTH, FT_DIR, SEEDS, MEAN, STD)  # noqa: E402
from models.MobileNetV3 import mobilenet_v3_large  # noqa: E402

OUT_JSON = os.path.join(REPO, "experiments", "r6_p0b_pooling.json")


def fov_mask(x, hw):
    """输入 [0,1] -> 组织区掩码 @hw(与 r6_p0_analysis 的 FOV 定义一致, 不腐蚀: 池化只需排除暗背景)。"""
    g = x.mean(1, keepdim=True)
    fov = (F.avg_pool2d(g, 15, 1, 7, count_include_pad=False) > 0.03).float()
    return F.adaptive_avg_pool2d(fov, hw)


@torch.no_grad()
def pooled_feats(model, X):
    acts = {}
    hs = [model.features[i].register_forward_hook(lambda m, inp, o, k=k: acts.__setitem__(k, o))
          for k, i in (("f12", 12), ("f16", 16))]
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
    out = {k: [] for k in ("f16_gap", "f12_gap", "f12_mgap", "f12_gem", "f12_max", "f12_mgem")}
    for i in range(0, len(X), 128):
        xb = X[i:i + 128].cuda()
        model((xb - mean) / std)
        a16, a12 = acts["f16"].float(), acts["f12"].float()
        m = fov_mask(xb, a12.shape[-2:])                       # (B,1,14,14) ∈[0,1]
        w = m / m.sum((2, 3), keepdim=True).clamp_min(1e-6)
        pos = a12.clamp_min(1e-6)                              # GeM 需非负(f12 为线性投影输出, 有负值)
        out["f16_gap"].append(a16.mean((2, 3)).cpu())
        out["f12_gap"].append(a12.mean((2, 3)).cpu())
        out["f12_mgap"].append((a12 * w).sum((2, 3)).cpu())
        out["f12_gem"].append(pos.pow(3).mean((2, 3)).pow(1 / 3).cpu())
        out["f12_max"].append(a12.amax((2, 3)).cpu())
        out["f12_mgem"].append((pos.pow(3) * w).sum((2, 3)).pow(1 / 3).cpu())
    for h in hs:
        h.remove()
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def main():
    splits = {sp: list_split(sp) for sp in ("train", "val", "test")}
    X = {sp: load_tensors(splits[sp][0]) for sp in splits}
    y = (splits["train"][1], splits["val"][1], splits["test"][1])
    feats = {}
    for name, seed in [("imnet", None)] + [(f"ft{s}", s) for s in SEEDS]:
        m = mobilenet_v3_large(num_classes=1000 if seed is None else 3)
        m.load_state_dict(torch.load(PTH if seed is None else os.path.join(FT_DIR.format(seed), "model_best.pth"),
                                     map_location="cpu"))
        m.eval().cuda()
        feats[name] = {sp: pooled_feats(m, X[sp]) for sp in X}
        del m
        torch.cuda.empty_cache()
    cfgs = {k: [k] for k in ("f12_gap", "f12_mgap", "f12_gem", "f12_max", "f12_mgem")}
    cfgs.update({f"f16+{k}": ["f16_gap", k] for k in ("f12_gap", "f12_mgap", "f12_gem", "f12_max", "f12_mgem")})
    jobs = [(n, c, tuple(np.concatenate([feats[n][sp][k] for k in parts], 1) for sp in ("train", "val", "test")))
            for n in feats for c, parts in cfgs.items()]
    outs = Parallel(n_jobs=12)(delayed(probe)(Xs, y, c) for _, c, Xs in jobs)
    res = {}
    for (n, c, _), o in zip(jobs, outs):
        res.setdefault(n, {})[c] = o
    print(f"{'probe':18s} | {'imnet f1/sensM/aucBM':24s} | {'ft(3s) f1/sensM/aucBM':24s}")
    for c in cfgs:
        im = res["imnet"][c]
        ft = [res[f"ft{s}"][c] for s in SEEDS]
        a = {k: np.mean([o[k] for o in ft]) for k in ("test_f1", "sens_m", "auc_bm")}
        print(f"{c:18s} | {im['test_f1']:.4f}/{im['sens_m']:.4f}/{im['auc_bm']:.4f} | "
              f"{a['test_f1']:.4f}/{a['sens_m']:.4f}/{a['auc_bm']:.4f}")
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    print("saved ->", OUT_JSON)


if __name__ == "__main__":
    main()
