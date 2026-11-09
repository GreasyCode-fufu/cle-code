#!/usr/bin/env python3
"""R6 实验 1/2 汇总(只读 evaluate.py 的 results.json; 同 seed 配对)。
实验 1: 主域 CLE 上 EPIC vs 最接近的微调策略方法(LP-FT / L2-SP / Surgical FT / cRT / τ-norm); 多配置方法另给
        "按 val 逐 seed 选配置"的结果(与 EPIC 的 C 选择同等调参预算)。
实验 2: ConvNeXt-T / StarNet-S2 / FasterNet-T0 上 EPIC 三组件消融 + 各主干门控决定。
用法: python experiments/r6_compare_collect.py [--json experiments/r6_compare_summary.json]
"""
import argparse
import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = "weights_cle_ac_v1_r6"
SEEDS = (42, 43, 44, 45, 46)
KEYS = [("macro_f1", "F1"), ("kappa", "κ"), ("mcc", "MCC"), ("auc_roc_macro", "AUROC")]


def runs(model, split="test"):
    out = {}
    for f in sorted(glob.glob(os.path.join(REPO, ROOT, f"{model}_seed*", model, "eval_*", "results.json")), reverse=True):
        r = json.load(open(f))
        s = int(f.split("_seed")[1].split("/")[0])
        if os.path.basename(r["test_dir"]) == split and s not in out:
            m = {k: r[k] * 100 for k, _ in KEYS}
            m["sensM"] = r["per_class"].get("Malignant_lesions", {}).get("sensitivity", np.nan) * 100
            out[s] = m
    return out


def val_selected(models):
    """逐 seed 在 models 中选 val Macro-F1 最高者, 返回其 test 指标。"""
    out, pick = {}, {}
    for s in SEEDS:
        cands = [(runs(m, "val").get(s, {}).get("macro_f1", -1), m) for m in models]
        best = max(cands)
        if best[0] < 0:
            continue
        t = runs(best[1]).get(s)
        if t:
            out[s], pick[s] = t, best[1]
    return out, pick


def row(name, r, ref, ref_name="baseline"):
    seeds = sorted(set(r) & set(SEEDS))
    if not seeds:
        return None
    vals = {k: np.array([r[s][k] for s in seeds]) for k in [k for k, _ in KEYS] + ["sensM"]}
    pair = [s for s in seeds if s in ref]
    d = np.array([r[s]["macro_f1"] - ref[s]["macro_f1"] for s in pair]) if pair else np.array([])
    return {"name": name, "n": len(seeds), **{k: round(float(v.mean()), 2) for k, v in vals.items()},
            "F1_std": round(float(vals["macro_f1"].std()), 2), f"dF1_vs_{ref_name}": round(float(d.mean()), 2) if len(d) else None,
            "up_seeds": f"{int((d > 0).sum())}/{len(d)}" if len(d) else ""}


def show(title, rows, refkey):
    print(f"\n### {title}")
    print(f"{'config':44s} n   F1(±std)        κ      MCC    AUROC  sensM   ΔF1   ↑seeds")
    for r in rows:
        if r is None:
            continue
        print(f"{r['name']:44s} {r['n']}  {r['macro_f1']:6.2f}±{r['F1_std']:4.2f}  {r['kappa']:6.2f} {r['mcc']:6.2f} "
              f"{r['auc_roc_macro']:6.2f} {r['sensM']:6.2f}  {r[refkey] if r[refkey] is not None else '':>6}  {r['up_seeds']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {}
    base = runs("mobilenet_v3_large")
    l2sp, l2sp_pick = val_selected(["mbv3_l2sp_a001", "mbv3_l2sp_a01"])
    sft, sft_pick = val_selected(["mbv3_sft_b1", "mbv3_sft_b2", "mbv3_sft_b3", "mbv3_sft_b4", "mbv3_sft_head"])
    e1 = [row("baseline MobileNetV3-L (full FT)", base, base),
          row("LP-FT (Kumar'22)", runs("mbv3_lpft"), base),
          row("L2-SP α=0.01 (Li'18, official default)", runs("mbv3_l2sp_a001"), base),
          row("L2-SP α=0.1", runs("mbv3_l2sp_a01"), base),
          row("L2-SP (α val-selected per seed)", l2sp, base),
          row("Surgical FT: block1 (/2–/4)", runs("mbv3_sft_b1"), base),
          row("Surgical FT: block2 (/8)", runs("mbv3_sft_b2"), base),
          row("Surgical FT: block3 (/16)", runs("mbv3_sft_b3"), base),
          row("Surgical FT: block4 (/32)", runs("mbv3_sft_b4"), base),
          row("Surgical FT: head only", runs("mbv3_sft_head"), base),
          row("Surgical FT (block val-selected per seed)", sft, base),
          row("cRT (Kang'20)", runs("mbv3_crt"), base),
          row("τ-norm (Kang'20)", runs("mbv3_taunorm"), base),
          row("EPIC ① only (freeze /2–/16, BN eval)", runs("mbv3_r6_fz16"), base),
          row("EPIC ①+②", runs("mbv3_fz16_dce16"), base),
          row("EPIC ①+②+③ (full)", runs("mbv3_fz16_dce4m"), base)]
    show("实验 1 — CLE 主域, 微调策略对比(5 seeds, Δ vs baseline 同 seed 配对)", e1, "dF1_vs_baseline")
    print("  val-selected picks: L2-SP", l2sp_pick, "| Surgical", sft_pick)
    epic = runs("mbv3_fz16_dce4m")
    print("  EPIC vs each (paired ΔF1, ↑seeds):")
    for r, m in [(runs("mbv3_lpft"), "LP-FT"), (l2sp, "L2-SP(sel)"), (sft, "Surgical(sel)"), (runs("mbv3_sft_b4"), "Surgical b4"),
                 (runs("mbv3_crt"), "cRT"), (runs("mbv3_taunorm"), "τ-norm")]:
        pair = sorted(set(r) & set(epic))
        if pair:
            d = np.array([epic[s]["macro_f1"] - r[s]["macro_f1"] for s in pair])
            print(f"    EPIC − {m:14s}: {d.mean():+.2f}  ({int((d > 0).sum())}/{len(d)} seeds)")
    out["exp1"] = {"rows": [r for r in e1 if r], "l2sp_pick": l2sp_pick, "sft_pick": sft_pick}
    # 实验 2
    out["exp2"] = {}
    for arch in ("convnext_tiny", "starnet_s2", "fast_t0"):
        b = runs(arch)
        rows = [row(f"{arch} baseline (full FT)", b, b), row("① fz3", runs(f"{arch}_fz3"), b),
                row("①+② (fz3, last-stage DCE)", runs(f"{arch}_fz3_dce_last"), b),
                row("①+②+③ (fz3, 4-tap FOV DCE) = EPIC", runs(f"{arch}_fz3_dce4m"), b),
                row("② only (full-FT, last-stage DCE)", runs(f"{arch}_dce_last"), b),
                row("②+③ (full-FT, 4-tap FOV DCE)", runs(f"{arch}_dce4m"), b),
                row("preserve-all LP arm + ②③", runs(f"{arch}_lp_dce4m"), b)]
        ftv = [v["macro_f1"] for v in runs(arch, "val").values()]
        lpv = [v["macro_f1"] for v in runs(f"{arch}_lp_dce4s", "val").values()]
        gate = None
        if ftv and lpv:
            gap = float(np.mean(ftv) - lpv[0])
            tier = "preserve-all" if gap <= 0 else ("①" if gap <= 1.5 else "full-FT")
            gate = {"LP_val": round(lpv[0], 2), "FT_val": round(float(np.mean(ftv)), 2), "gap": round(gap, 2), "tier": tier}
        show(f"实验 2 — {arch}（5 seeds; LP 臂为确定性单值）  gate: {gate}", rows, "dF1_vs_baseline")
        out["exp2"][arch] = {"rows": [r for r in rows if r], "gate": gate}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
