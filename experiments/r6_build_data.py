#!/usr/bin/env python3
"""EPIC-MobileNet(暂定名) 发布包 experiment_data 生成脚本(数据溯源)。
在实验仓库根目录运行: python experiments/r6_build_data.py <输出目录>
全部数字直接读取 evaluate.py 输出的 results.json(及 r6_* 分析 json), 不经手工转录。
生成:
  cle_main_results.csv          主域消融链 7 配置 × 5 seeds × 9 指标 + 逐类敏感度 + MEAN/STD 行
  cle_comparison_backbones.csv  轻量主干对比(5 seeds 均值±std)
  crossdomain_ablation.csv      5 域 × 配置 × seed 的 Macro-F1/κ/AUROC
  transferability_gate.csv      门控: LP_val / FT_val / gap / 决定 / 观察
  significance.csv              主域逐 seed 配对 bootstrap CI + McNemar + 跨 seed 配对 t
  error_structure.csv           主域 5 seeds 按亚型×方向错误数(baseline vs 最终网络)
  negative_screening_summary.csv 未通过检验的候选汇总
  model_complexity.csv          params/MACs/CPU/GPU 延迟(r6_bench_latency.json)
  run_meta/                     最终网络 5 seeds 阶段 1 + 阶段 2 元数据(含预训练严格加载审计与重拟合审计)
"""
import csv
import glob
import json
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "experiments", "r6_experiment_data")
M9 = [("macro_f1", "Macro-F1"), ("accuracy", "Accuracy"), ("weighted_f1", "Weighted-F1"),
      ("balanced_accuracy", "BalAcc"), ("macro_sensitivity", "mSens"), ("macro_specificity", "mSpec"),
      ("kappa", "Kappa"), ("mcc", "MCC"), ("auc_roc_macro", "AUROC")]
CLE_CHAIN = [("baseline (MobileNetV3-L)", "mobilenet_v3_large"),
             ("+① evidence-preserving freeze", "mbv3_r6_fz16"),
             ("+①+② decoupled convex head (f16)", "mbv3_fz16_dce16"),
             ("+①+② (4-scale GAP)", "mbv3_fz16_dce4s"),
             ("+①+②+③ = EPIC-MobileNet", "mbv3_fz16_dce4m"),
             ("② only (convex head on full-FT backbone)", "mbv3_dce16"),
             ("②+③ (no ①)", "mbv3_dce4m")]
BACKBONES = [("MobileNetV3-L", "mobilenet_v3_large"), ("StarNet-S2 (CVPR'24)", "starnet_s2"),
             ("MobileNetV4-Conv-S (ECCV'24)", "mobilenetv4_conv_small"), ("FasterNet-T0 (CVPR'23)", "fast_t0"),
             ("GhostNet", "ghostnet"), ("ShuffleNetV2 x1.0", "shufflenet_v2_x1_0"),
             ("EPIC-MobileNet (ours)", "mbv3_fz16_dce4m")]
DOMAINS = [("busi", "BUSI"), ("heel", "heelBone"), ("covid", "covid"), ("cub", "CUB-200"), ("tcga", "TCGA")]
XD_CFG = [("baseline", "mobilenet_v3_large"), ("①", "mbv3_r6_fz16"), ("①+②", "mbv3_fz16_dce16"),
          ("①+②+③", "mbv3_fz16_dce4m"), ("②+③", "mbv3_dce4m"), ("LP-arm+②③", "mbv3_lp_dce4m")]


def test_results(root, model):
    out = {}
    for f in glob.glob(os.path.join(REPO, root, f"{model}_seed*", model, "eval_*", "results.json")):
        r = json.load(open(f))
        if os.path.basename(r.get("test_dir", "")) == "test":
            s = int(f.split("_seed")[1].split("/")[0])
            if s not in out or f > out[s][0]:          # 同 seed 多个 eval 取最新
                out[s] = (f, r)
    return {s: v[1] for s, v in sorted(out.items())}


def w(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        cw = csv.writer(fh)
        cw.writerow(header)
        cw.writerows(rows)
    print(f"  {name}: {len(rows)} rows")


def main():
    os.makedirs(OUT, exist_ok=True)
    # 1. 主域消融链
    rows = []
    for label, model in CLE_CHAIN:
        runs = test_results("weights_cle_ac_v1_r6", model)
        vals = []
        for s, r in runs.items():
            per = r["per_class"]
            v = [round(r[k] * 100, 2) for k, _ in M9] + [round(per[c]["sensitivity"] * 100, 2) for c in per]
            vals.append(v)
            rows.append([label, model, s] + v)
        a = np.array(vals)
        rows.append([label, model, "MEAN"] + list(np.round(a.mean(0), 2)))
        rows.append([label, model, "STD"] + list(np.round(a.std(0), 2)))
    w("cle_main_results.csv", ["config", "registry", "seed"] + [n for _, n in M9] + ["sens_Benign", "sens_Malignant", "sens_Normal"], rows)
    # 2. 轻量主干对比
    bench = json.load(open(os.path.join(REPO, "experiments", "r6_bench_latency.json")))
    rows = []
    for label, model in BACKBONES:
        runs = test_results("weights_cle_ac_v1_r6", model)
        f1 = np.array([r["macro_f1"] * 100 for r in runs.values()])
        kap = np.array([r["kappa"] * 100 for r in runs.values()])
        auc = np.array([r["auc_roc_macro"] * 100 for r in runs.values()])
        b = bench.get(model if model != "mbv3_fz16_dce4m" else "mbv3_dce4m", {})
        rows.append([label, model, len(f1), f"{f1.mean():.2f}±{f1.std():.2f}", f"{kap.mean():.2f}", f"{auc.mean():.2f}",
                     b.get("params_M"), b.get("macs_G"), b.get("cpu_t1_ms"), b.get("gpu_b1_ms"), b.get("gpu_b64_amp_img_s")])
    w("cle_comparison_backbones.csv", ["model", "registry", "n_seeds", "Macro-F1", "Kappa", "AUROC", "params_M", "MACs_G",
                                       "cpu_1thread_ms", "gpu_bs1_ms", "gpu_bs64_amp_img_per_s"], rows)
    # 3. 跨域消融
    rows = []
    for dom, nice in DOMAINS:
        for cfg, model in XD_CFG:
            for s, r in test_results(f"weights_xd_{dom}_r6", model).items():
                rows.append([nice, cfg, model, s, round(r["macro_f1"] * 100, 2), round(r["kappa"] * 100, 2),
                             round(r["auc_roc_macro"] * 100, 2)])
    w("crossdomain_ablation.csv", ["domain", "config", "registry", "seed", "Macro-F1", "Kappa", "AUROC"], rows)
    # 4. 门控
    probe = json.load(open(os.path.join(REPO, "experiments", "r6_transfer_probe.json")))
    rows = []
    for dom, nice, root, res in [("cle", "CLE (main)", "weights_cle_ac_v1_r6", "experiments/results_cle_ac_v1_r6.csv")] + \
            [(d, n, f"weights_xd_{d}_r6", f"experiments/results_xd_{d}_r6.csv") for d, n in DOMAINS]:
        if dom not in probe or not os.path.exists(os.path.join(REPO, res)):
            continue
        ft = [float(r["val_f1"]) * 100 for r in csv.DictReader(open(os.path.join(REPO, res)))
              if r["model"] == "mobilenet_v3_large" and r["status"] == "ok"]
        gap = np.mean(ft) - probe[dom]["probe_val_f1"]
        mean = lambda d: np.mean([r["macro_f1"] * 100 for r in d.values()]) if d else float("nan")  # noqa: E731
        tiers = {"preserve-all": mean(test_results(root, "mbv3_lp_dce4m")), "freeze-/16 (①)": mean(test_results(root, "mbv3_fz16_dce4m")),
                 "full-FT": mean(test_results(root, "mbv3_dce4m"))}
        pick = "preserve-all" if gap <= 0 else ("freeze-/16 (①)" if gap <= 1.5 else "full-FT")
        best = max(tiers, key=lambda k: tiers[k] if tiers[k] == tiers[k] else -1)
        base = mean(test_results(root, "mobilenet_v3_large"))
        rows.append([nice, probe[dom]["probe_val_f1"], round(np.mean(ft), 2), round(gap, 2), pick, round(tiers[pick], 2), best,
                     round(tiers[best], 2), round(tiers["preserve-all"], 2), round(tiers["freeze-/16 (①)"], 2), round(tiers["full-FT"], 2),
                     round(base, 2), round(tiers[pick] - base, 2), "yes" if tiers[best] - tiers[pick] < 0.3 else "no"])
    w("transferability_gate.csv", ["domain", "LP_val_F1", "FT_val_F1", "gap", "gate_v2_pick", "test_F1_pick", "best_tier",
                                   "test_F1_best", "test_preserve_all", "test_freeze16", "test_fullFT", "test_baseline",
                                   "pick_minus_baseline", "pick_within_0.3_of_best"], rows)
    # 5. 显著性与误差结构
    sig = json.load(open(os.path.join(REPO, "experiments", "r6_significance.json")))
    rows = [[s, v["f1_a"], v["f1_b"], v["dF1"], v["dF1_CI95"][0], v["dF1_CI95"][1], v["mcnemar_b_fixes"],
             v["mcnemar_b_breaks"], f"{v['mcnemar_p']:.2e}"] for s, v in sig["per_seed"].items()]
    pt = sig["paired_t"]
    rows.append(["paired-t(n=%d)" % pt["n"], "", "", pt["mean_dF1"], "", "", "", "", f"t={pt['t']:.2f}, p={pt['p']:.2e}"])
    w("significance.csv", ["seed", "F1_baseline", "F1_EPIC", "dF1", "CI95_lo", "CI95_hi", "fixes", "breaks", "McNemar_p"], rows)
    w("error_structure.csv", ["error(true:subtype->pred)", "baseline_5seeds", "EPIC_5seeds"],
      [[k, v["a"], v["b"]] for k, v in sig["errors_5seeds"].items()])
    # 6. 负结果
    neg = [("star2 cross-scale star head (StarNet CVPR'24)", "mbv3_r6_star2", "mobilenet_v3_large"),
           ("star3 (+/8)", "mbv3_r6_star3", "mobilenet_v3_large"), ("star1 same-scale", "mbv3_r6_star1", "mobilenet_v3_large"),
           ("star2x expanded-space evidence", "mbv3_r6_star2x", "mobilenet_v3_large"),
           ("mlp256 shrunk head", "mbv3_r6_mlp256", "mobilenet_v3_large"), ("linear head (end-to-end)", "mbv3_r6_lin", "mobilenet_v3_large"),
           ("LSConv@/16 backbone (LSNet CVPR'25)", "mbv3_r6_s2_lsbb", "mbv3_r6_star2"),
           ("LSConv@/16 side path", "mbv3_r6_s2_lsside", "mbv3_r6_star2"),
           ("IDC@/16 (InceptionNeXt CVPR'24)", "mbv3_r6_s2_idcbb", "mbv3_r6_star2"),
           ("/16 pretrained anchor λ=1", "mbv3_r6_s2_anc1", "mbv3_r6_star2"), ("/16 pretrained anchor λ=0.3", "mbv3_r6_s2_anc03", "mbv3_r6_star2"),
           ("MatchED-style edge prior (CVPR'26)", "mbv3_r6_s2_edge", "mbv3_r6_star2"),
           ("BN-adaptive freeze /2–/16 (stage 1)", "mbv3_r6_fz16bn", "mobilenet_v3_large"),
           ("BN-adaptive freeze + DCE4m", "mbv3_fz16bn_dce4m", "mobilenet_v3_large")]
    rows = []
    for label, model, ref in neg:
        r, b = test_results("weights_cle_ac_v1_r6", model), test_results("weights_cle_ac_v1_r6", ref)
        pair = sorted(set(r) & set(b))
        if not pair:
            continue
        d = [r[s]["macro_f1"] * 100 - b[s]["macro_f1"] * 100 for s in pair]
        rows.append([label, model, ref, len(pair), round(np.mean([r[s]["macro_f1"] * 100 for s in pair]), 2), round(np.mean(d), 2),
                     "rejected"])
    w("negative_screening_summary.csv", ["candidate", "registry", "paired_reference", "n_seeds", "mean_F1", "paired_dF1", "verdict"], rows)
    # 7. 复杂度
    w("model_complexity.csv", ["registry"] + list(next(iter(bench.values())).keys()),
      [[k] + list(v.values()) for k, v in bench.items()])
    # 8. run_meta
    rm = os.path.join(OUT, "run_meta")
    os.makedirs(rm, exist_ok=True)
    for s in (42, 43, 44, 45, 46):
        for model, tag in (("mbv3_r6_fz16", "stage1"), ("mbv3_fz16_dce4m", "stage2_refit")):
            src = os.path.join(REPO, "weights_cle_ac_v1_r6", f"{model}_seed{s}", model, "run_meta.json")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(rm, f"{tag}_seed{s}.json"))
    print("run_meta copied ->", rm)


if __name__ == "__main__":
    main()
