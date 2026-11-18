#!/usr/bin/env python3
"""分支解锁探针: 证明修复后的 SCG / HSC 已脱离零初始化, 真正参与前向。

修复后 SCG 与 HSC 不再使用外部标量门(冒烟实测 scalar_gates=0), 其"初始恒等"由末层
零初始化实现: SCG 的 `scg.c3`、HSC 的 `hsc.fc2`。因此判断模块是否被训练激活,
必须看这两类权重的范数是否显著 > 0; 旧脚本只看 egb_g/scg_g/hsc.g/esa_g 会漏判。

用法:
    python probe_branch.py weights_cle_fixed_v1
"""
import glob
import json
import os
import sys

import torch

ROOT = sys.argv[1] if len(sys.argv) > 1 else "weights_cle_fixed_v1"
REPO = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(ROOT):
    ROOT = os.path.join(REPO, ROOT)

GATE_SUFFIX = ("egb_g", "scg_g", "hsc.g", "esa_g")
ZERO_KEYS = ("scg.c3.weight", "hsc.fc2.weight")


def unwrap(sd):
    for k in ("state_dict", "model", "net", "model_state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            return sd[k]
    return sd


def strip_prefix(k):
    for pref in ("module.", "model.", "net."):
        if k.startswith(pref):
            return k[len(pref):]
    return k


rows = []
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d)
    if not os.path.isdir(p):
        continue
    ck = sorted(glob.glob(os.path.join(p, "*", "model_best.pth")))
    if not ck:
        continue
    try:
        sd = unwrap(torch.load(ck[0], map_location="cpu"))
    except Exception as e:
        rows.append({"run": d, "error": str(e)[:100]})
        continue

    gates = {}
    zero = {}
    for k, v in sd.items():
        kk = strip_prefix(k)
        if kk.endswith(GATE_SUFFIX):
            try:
                gates[kk] = float(v.reshape(-1)[0])
            except Exception:
                pass
        for zk in ZERO_KEYS:
            if kk.endswith(zk):
                w = v.float()
                zero.setdefault(zk, []).append({
                    "key": kk,
                    "l2": float(w.norm()),
                    "max": float(w.abs().max()),
                    "mean": float(w.abs().mean()),
                    "shape": list(w.shape),
                })

    rows.append({
        "run": d,
        "gates": {
            "n": len(gates),
            "min": min(gates.values()) if gates else None,
            "max": max(gates.values()) if gates else None,
            "vals_neg": sum(1 for v in gates.values() if v < 0),
            "abs_gt_1e-2": sum(1 for v in gates.values() if abs(v) > 1e-2),
        },
        "zero": {zk: {
            "n": len(v),
            "l2_min": min(x["l2"] for x in v),
            "l2_max": max(x["l2"] for x in v),
            "absmax": max(x["max"] for x in v),
            "still_dead": sum(1 for x in v if x["l2"] == 0.0),
        } for zk, v in zero.items()},
    })

print(json.dumps(rows, indent=1, ensure_ascii=False))

# ── 判定摘要 ──
print("\n=== 判定 ===")
print(f"{'run':32s} {'gates':>5s} {'|g|max':>9s} {'>1e-2':>6s} "
      f"{'scg.c3 L2(max)':>15s} {'dead':>5s} {'hsc.fc2 L2(max)':>16s} {'dead':>5s}")
for r in rows:
    if "error" in r:
        print(f"{r['run']:32s} ERROR {r['error']}")
        continue
    g = r["gates"]
    sc = r["zero"].get("scg.c3.weight")
    hs = r["zero"].get("hsc.fc2.weight")
    gmax = f"{g['max']:.4e}" if g["max"] is not None else "-"
    scv = f"{sc['l2_max']:.4e}" if sc else "-"
    scd = str(sc["still_dead"]) if sc else "-"
    hsv = f"{hs['l2_max']:.4e}" if hs else "-"
    hsd = str(hs["still_dead"]) if hs else "-"
    print(f"{r['run']:32s} {g['n']:5d} {gmax:>9s} {g['abs_gt_1e-2']:6d} "
          f"{scv:>15s} {scd:>5s} {hsv:>16s} {hsd:>5s}")
