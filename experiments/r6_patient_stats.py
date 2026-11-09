#!/usr/bin/env python3
"""R6 实验 3: 患者级统计 —— 同一患者多帧图像相关, 图像级检验会偏乐观; 本脚本以患者为重采样单元。
患者映射: experiments/cle_ac_v1_patient_groups.csv(来自 GPT 项目 ac_dataset_v1/dataset_manifest.json 的 group_id,
已按 sha256 + split 与本仓库 3080 张图逐一核对一致; test 41 位患者, 每人 1–109 张, 各划分间无患者重叠)。

对每对模型 (A=参照, B=比较), 使用 evaluate.py 输出的逐图预测(predictions.csv, 各 seed 最新 test 评测):
  1) 患者聚类 bootstrap(B=5000): 有放回重采样 41 位患者, 拼接其全部图像, 计算 Macro-F1 / κ / 恶性敏感度之差;
     同一重采样同时作用于 5 个 seed, 给出逐 seed 与 5-seed 平均差的 95% CI 及单侧 p=P(Δ≤0);
     并与图像级 bootstrap(重采样图像)对比 CI 宽度。
  2) 患者级配对检验: 每位患者的准确率差(5 seeds 平均)做 Wilcoxon 符号秩检验; 统计改善/变差/持平的患者数。
  3) 留一患者: 逐一剔除患者后 5-seed 平均 ΔF1 的范围(检查单个大患者的影响)。
用法: python experiments/r6_patient_stats.py --pairs mobilenet_v3_large:mbv3_fz16_dce4m [A:B ...]
输出: experiments/r6_patient_stats.json(按模型对合并写入)
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]
OUT = os.path.join(REPO, "experiments", "r6_patient_stats.json")


def preds(root, model, seed):
    for f in sorted(glob.glob(os.path.join(REPO, root, f"{model}_seed{seed}", model, "eval_*", "results.json")), reverse=True):
        if os.path.basename(json.load(open(f))["test_dir"]) == "test":
            rows = csv.DictReader(open(os.path.join(os.path.dirname(f), "predictions.csv")))
            return {"S_" + r["image_path"].rsplit("_S_", 1)[1].split(".")[0]: (C.index(r["true_label"]), C.index(r["pred_label"]))
                    for r in rows}
    raise FileNotFoundError(f"{model} seed{seed}")


def metrics(y, p, k=3):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(0), 1e-12)
    rec = tp / np.maximum(cm.sum(1), 1e-12)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    n = cm.sum()
    po, pe = tp.sum() / n, (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return np.array([f1.mean(), (po - pe) / (1 - pe), rec[1]]) * 100   # Macro-F1, κ, 恶性敏感度


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["mobilenet_v3_large:mbv3_fz16_dce4m"])
    ap.add_argument("--root", default="weights_cle_ac_v1_r6")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--B", type=int, default=5000)
    a = ap.parse_args()
    grp = {r["sample_id"]: r["group_id"] for r in csv.DictReader(open(os.path.join(REPO, "experiments", "cle_ac_v1_patient_groups.csv")))
           if r["split"] == "test"}
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    names = ("Macro-F1", "Kappa", "sens_M")
    for pair in a.pairs:
        A, Bm = pair.split(":")
        pa = {s: preds(a.root, A, s) for s in a.seeds}
        pb = {s: preds(a.root, Bm, s) for s in a.seeds}
        ids = sorted(pa[a.seeds[0]])
        assert set(ids) == set(grp), "test 图像与患者映射不一致"
        y = np.array([pa[a.seeds[0]][i][0] for i in ids])
        YA = np.stack([[pa[s][i][1] for i in ids] for s in a.seeds])
        YB = np.stack([[pb[s][i][1] for i in ids] for s in a.seeds])
        pats = sorted(set(grp.values()))
        pidx = [np.array([j for j, i in enumerate(ids) if grp[i] == p]) for p in pats]
        obs = np.array([metrics(y, YB[k]) - metrics(y, YA[k]) for k in range(len(a.seeds))])   # (seeds, 3)
        rng = np.random.default_rng(0)
        boot_pat = np.empty((a.B, len(a.seeds), 3))
        boot_img = np.empty((a.B, len(a.seeds), 3))
        for b in range(a.B):
            sel = np.concatenate([pidx[i] for i in rng.integers(0, len(pats), len(pats))])
            imgs = rng.integers(0, len(y), len(y))
            for k in range(len(a.seeds)):
                boot_pat[b, k] = metrics(y[sel], YB[k][sel]) - metrics(y[sel], YA[k][sel])
                boot_img[b, k] = metrics(y[imgs], YB[k][imgs]) - metrics(y[imgs], YA[k][imgs])
        rec = {"n_patients": len(pats), "n_images": int(len(y)), "observed_mean": dict(zip(names, np.round(obs.mean(0), 2).tolist()))}
        for j, n in enumerate(names):
            mp, mi = boot_pat[:, :, j].mean(1), boot_img[:, :, j].mean(1)
            rec[n] = {"mean_delta": round(float(obs[:, j].mean()), 2),
                      "patient_CI95": [round(float(v), 2) for v in np.percentile(mp, [2.5, 97.5])],
                      "patient_p_one_sided": float((mp <= 0).mean()),
                      "image_CI95": [round(float(v), 2) for v in np.percentile(mi, [2.5, 97.5])],
                      "per_seed_patient_CI95": {s: [round(float(v), 2) for v in np.percentile(boot_pat[:, k, j], [2.5, 97.5])]
                                                for k, s in enumerate(a.seeds)}}
        # 患者级配对检验(每位患者准确率差, 5 seeds 平均)
        accA = np.array([(YA[:, ix] == y[ix]).mean() for ix in pidx])
        accB = np.array([(YB[:, ix] == y[ix]).mean() for ix in pidx])
        d = accB - accA
        nz = d[np.abs(d) > 1e-12]
        w = wilcoxon(nz) if len(nz) else None
        rec["patient_accuracy_wilcoxon"] = {"improved": int((d > 1e-12).sum()), "worsened": int((d < -1e-12).sum()),
                                            "tied": int((np.abs(d) <= 1e-12).sum()),
                                            "statistic": float(w.statistic) if w else None, "p_two_sided": float(w.pvalue) if w else None}
        # 留一患者
        lopo = []
        for i, ix in enumerate(pidx):
            keep = np.setdiff1d(np.arange(len(y)), ix)
            lopo.append(np.mean([metrics(y[keep], YB[k][keep])[0] - metrics(y[keep], YA[k][keep])[0] for k in range(len(a.seeds))]))
        big = int(np.argmax([len(ix) for ix in pidx]))
        rec["leave_one_patient_out_dF1"] = {"min": round(float(min(lopo)), 2), "max": round(float(max(lopo)), 2),
                                            "drop_largest_patient": round(float(lopo[big]), 2), "largest_patient_images": int(len(pidx[big]))}
        res[pair] = rec
        print(f"\n== {A} -> {Bm}  ({len(pats)} patients, {len(y)} images, {len(a.seeds)} seeds)")
        for n in names:
            r = rec[n]
            print(f"  Δ{n:8s} {r['mean_delta']:+6.2f}  patient-cluster 95% CI [{r['patient_CI95'][0]:+.2f}, {r['patient_CI95'][1]:+.2f}] "
                  f"p1={r['patient_p_one_sided']:.4f}   (image-level CI [{r['image_CI95'][0]:+.2f}, {r['image_CI95'][1]:+.2f}])")
        wx = rec["patient_accuracy_wilcoxon"]
        print(f"  patients improved/worsened/tied {wx['improved']}/{wx['worsened']}/{wx['tied']}, Wilcoxon p={wx['p_two_sided']}")
        lo = rec["leave_one_patient_out_dF1"]
        print(f"  leave-one-patient-out ΔF1 range [{lo['min']:+.2f}, {lo['max']:+.2f}]; drop largest patient ({lo['largest_patient_images']} imgs) -> {lo['drop_largest_patient']:+.2f}")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
