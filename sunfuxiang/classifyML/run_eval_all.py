#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量评估 + 汇总对比脚本
======================
遍历  3 seed(42/43/44) × 2 数据集(hq/full) × 6 模型 = 36 组合,
对每个调用 evaluate.py 做测试集评估(归一化自动读 run_meta.json),
收集核心指标,汇总成三张表:

  summary_hq.csv     : hq 数据集,每行一个 (模型, seed) 的全部指标
  summary_full.csv   : full 数据集,同样
  summary_compare.csv: 核心对比表,每行一个模型,
                        按 3 个 seed 聚合 mean±std,hq 与 full 并列(差值=hq−full)
                        macro_f1 作为主指标放最前。

用法:
  python run_eval_all.py                # 跑缺失的评估 + 汇总(默认)
  python run_eval_all.py --dry-run      # 只打印计划(谁跳过/谁要跑),不评估
  python run_eval_all.py --summary-only # 不跑评估,只用已有 results.json 汇总
  python run_eval_all.py --force        # 强制重跑全部 36 个(忽略已有结果)
  python run_eval_all.py --device cuda:1 --batch-size 128   # 透传给 evaluate.py
"""

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

# ============================ 配置(硬编码矩阵)============================
REPO = Path(__file__).resolve().parent
EVAL_PY = REPO / "evaluate.py"
TEST_PATH = "/home/login/cle/sunfuxiang/datasets/3class/test"

DATASETS = ["hq", "full"]
SEEDS = [42, 43, 44]
MODELS = [
    "fast_t0",
    "resnet34",
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "pvt_tiny",
    "repvit_m1",
]
# 汇总用的核心指标(macro_f1 为主指标,故放第一)
METRICS = ["macro_f1", "accuracy", "auc_roc_macro",
           "macro_sensitivity", "macro_specificity", "weighted_f1"]
PRIMARY = "macro_f1"


# ============================ 路径辅助 ============================

def weights_dir(ds, seed):
    return REPO / f"weights_3class_{ds}_seed{seed}"


def model_dir(ds, seed, m):
    return weights_dir(ds, seed) / m


def best_weights(ds, seed, m):
    return model_dir(ds, seed, m) / f"{m}_3class_{ds}_seed{seed}_best.pth"


def find_done_results(ds, seed, m):
    """返回该组合下『字段齐全』的 results.json,按时间戳 新→旧 排序。

    - evaluate.py 输出在 <model子目录>/eval_<时间戳>/results.json,
      时间戳目录按字典序 = 时间序,reverse 后最新在最前。
    - 只保留含全部 METRICS 字段的(eval.py 旧版可能缺字段),取最新即可。
    """
    out = []
    for f in glob.glob(str(model_dir(ds, seed, m) / "eval_*/results.json")):
        try:
            d = json.load(open(f, "r", encoding="utf-8"))
        except Exception:
            continue
        if all(k in d for k in METRICS):
            out.append(f)
    out.sort(reverse=True)  # eval_<ts> 字典序倒序 = 最新优先
    return out


def load_metrics(path):
    d = json.load(open(path, "r", encoding="utf-8"))
    return {k: float(d[k]) for k in METRICS}


# ============================ 评估执行 ============================

def run_eval(ds, seed, m, data_path, device=None, batch_size=None):
    """调用 evaluate.py。只传 --model/--weights/--data-path,
    归一化让其自动读 weights 旁 run_meta.json(不手动传)。"""
    w = best_weights(ds, seed, m)
    cmd = [sys.executable, str(EVAL_PY),
           "--model", m,
           "--weights", str(w),
           "--data-path", data_path]
    if device:
        cmd += ["--device", device]
    if batch_size:
        cmd += ["--batch-size", str(batch_size)]
    print(f"\n>>> [{ds} seed{seed} {m}] 评估中 ...")
    print("    $ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(REPO))
    return r.returncode == 0


# ============================ 统计辅助 ============================

def mean_std(vals):
    """3 个 seed 的样本均值/标准差(ddof=1)。"""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def fmt_ms(vals):
    m, s = mean_std(vals)
    if m is None:
        return "—"
    return f"{m:.4f}±{s:.4f}"


def fmt_diff(diff):
    if diff is None:
        return "—"
    return f"{diff:+.4f}"


# ============================ 汇总:明细 CSV ============================

def write_detail_csv(path, ds, results):
    """results: {(ds,seed,model): {metric: value}}; 每行一个 (模型, seed)。"""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "seed"] + METRICS)
        rows = []
        for (d, seed, m), met in results.items():
            if d != ds:
                continue
            row = [ds, m, seed] + [f"{met[k]:.6f}" if met and k in met else "" for k in METRICS]
            rows.append((m, seed, row))
        rows.sort(key=lambda x: (MODELS.index(x[0]) if x[0] in MODELS else 999, x[1]))
        for _, _, row in rows:
            w.writerow(row)
    print(f"  [写出] {path}")


# ============================ 汇总:对比 CSV ============================

def write_compare_csv(path, agg):
    """agg[model][ds] = {metric: (mean, std)};每行一个模型。
    每个指标输出 hq_mean±std / full_mean±std / diff(hq−full) 三列,
    macro_f1 在最前。"""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        header = ["model"]
        for met in METRICS:
            header += [f"{met}__hq", f"{met}__full", f"{met}__diff(hq-full)"]
        w.writerow(header)
        for m in MODELS:
            row = [m]
            hq = agg.get(m, {}).get("hq", {})
            fl = agg.get(m, {}).get("full", {})
            for met in METRICS:
                hm, hs = hq.get(met, (None, None))
                fm, fs = fl.get(met, (None, None))
                hq_s = f"{hm:.4f}±{hs:.4f}" if hm is not None else ""
                fl_s = f"{fm:.4f}±{fs:.4f}" if fm is not None else ""
                diff = (hm - fm) if (hm is not None and fm is not None) else None
                row += [hq_s, fl_s, f"{diff:+.4f}" if diff is not None else ""]
            w.writerow(row)
    print(f"  [写出] {path}")


# ============================ 终端对比表 ============================

def print_compare_table(agg):
    W = 92
    print("\n" + "=" * W)
    print("  核心对比表(mean±std,聚合 3 个 seed;Δ = hq − full,正值=高质量集更好)")
    print("=" * W)

    for met in METRICS:
        tag = "  ← 主指标" if met == PRIMARY else ""
        print(f"\n  【{met}】{tag}")
        print(f"  {'model':<30}| {'hq(高质量集)':<18}| {'full(全量)':<18}| {'Δ(hq−full)':<12}")
        print(f"  {'-'*30}+{'-'*18}+{'-'*18}+{'-'*12}")
        for m in MODELS:
            hq = agg.get(m, {}).get("hq", {}).get(met, (None, None))
            fl = agg.get(m, {}).get("full", {}).get(met, (None, None))
            hm, _ = hq
            fm, _ = fl
            diff = (hm - fm) if (hm is not None and fm is not None) else None
            flag = ""
            if diff is not None:
                flag = "  hq更优" if diff > 0 else ("  full更优" if diff < 0 else "  持平")
            print(f"  {m:<30}| {fmt_ms_list(hq):<18}| {fmt_ms_list(fl):<18}| "
                  f"{fmt_diff(diff):<12}{flag}")

    # 主指标排名速览
    print("\n  " + "-" * W)
    print("  【主指标 macro_f1 排名速览】(按 hq 的 mean 降序)")
    ranked = sorted(
        MODELS,
        key=lambda m: agg.get(m, {}).get("hq", {}).get(PRIMARY, (-1,))[0],
        reverse=True,
    )
    for i, m in enumerate(ranked, 1):
        hm = agg.get(m, {}).get("hq", {}).get(PRIMARY, (None,))[0]
        fm = agg.get(m, {}).get("full", {}).get(PRIMARY, (None,))[0]
        if hm is None and fm is None:
            continue
        winner = "—" if (hm is None or fm is None) else (
            "hq" if hm > fm else ("full" if fm > hm else "持平"))
        print(f"    {i}. {m:<30} hq={hm if hm is None else f'{hm:.4f}'}  "
              f"full={fm if fm is None else f'{fm:.4f}'}  → 更优:{winner}")
    print("=" * W)


def fmt_ms_list(ms):
    """ms=(mean,std) 或 None → '0.9044±0.0032'。"""
    if ms is None or ms[0] is None:
        return "—"
    return f"{ms[0]:.4f}±{ms[1]:.4f}"


# ============================ 主流程 ============================

def main():
    ap = argparse.ArgumentParser(description="36 组合批量评估 + hq/full 对比汇总")
    ap.add_argument("--data-path", default=TEST_PATH, help="测试集目录(默认 3class/test)")
    ap.add_argument("--out-dir", default=str(REPO), help="汇总 CSV 输出目录(默认仓库根)")
    ap.add_argument("--device", default=None, help="透传 evaluate.py 的 --device(默认不传)")
    ap.add_argument("--batch-size", type=int, default=None, help="透传 evaluate.py 的 --batch-size")
    ap.add_argument("--summary-only", action="store_true", help="不跑评估,仅用已有结果汇总")
    ap.add_argument("--force", action="store_true", help="忽略已有结果,强制重跑全部 36 个")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划,不评估不汇总")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 构造 36 组合并分类:已就绪 vs 待评估 ----
    combos = [(ds, seed, m) for ds in DATASETS for seed in SEEDS for m in MODELS]
    done, todo = [], []
    for ds, seed, m in combos:
        ready = best_weights(ds, seed, m).is_file()
        if not ready:
            print(f"  [警告] 权重缺失,跳过: {best_weights(ds, seed, m)}")
            continue
        exists = (not args.force) and len(find_done_results(ds, seed, m)) > 0
        (done if exists else todo).append((ds, seed, m))

    print(f"\n计划: 共 {len(combos)} 组合 | 已就绪(跳过){len(done)} | 待评估 {len(todo)}")
    if args.force:
        print("  (--force: 全部重跑)")
    if args.dry_run:
        print("\n[dry-run] 待评估列表:")
        for ds, seed, m in todo:
            print(f"    - {ds} seed{seed} {m}")
        print("\n[dry-run] 将跳过(已有齐全结果):")
        for ds, seed, m in done:
            print(f"    - {ds} seed{seed} {m}")
        print("\n[dry-run] 未执行评估与汇总。")
        return

    # ---- 2. 执行评估(仅 todo)----
    if args.summary_only:
        print("\n--summary-only: 跳过评估,直接用已有结果汇总。")
    else:
        for i, (ds, seed, m) in enumerate(todo, 1):
            print(f"\n===== [{i}/{len(todo)}] =====")
            ok = run_eval(ds, seed, m, args.data_path, args.device, args.batch_size)
            if not ok:
                print(f"    [错误] evaluate.py 返回非 0:{ds} seed{seed} {m} (继续下一个)")
            elif not find_done_results(ds, seed, m):
                print(f"    [警告] 评估结束但未找到齐全的 results.json:{ds} seed{seed} {m}")

    # ---- 3. 收集全部 36 组合的指标(取最新齐全 results.json)----
    results = {}   # (ds,seed,model) -> {metric: value}
    missing = []
    for ds, seed, m in combos:
        cands = find_done_results(ds, seed, m)
        if not cands:
            missing.append((ds, seed, m))
            results[(ds, seed, m)] = None
            continue
        results[(ds, seed, m)] = load_metrics(cands[0])  # 最新

    if missing:
        print("\n[警告] 以下组合仍缺少齐全的 results.json(汇总中按缺失处理):")
        for ds, seed, m in missing:
            print(f"    - {ds} seed{seed} {m}")

    # ---- 4. 写明细 CSV(hq / full 分开)----
    print("\n[汇总] 写出明细表 ...")
    write_detail_csv(out_dir / "summary_hq.csv", "hq", results)
    write_detail_csv(out_dir / "summary_full.csv", "full", results)

    # ---- 5. 按模型聚合 3 seed → mean±std ----
    agg = {}  # model -> ds -> metric -> (mean, std)
    for m in MODELS:
        agg[m] = {}
        for ds in DATASETS:
            agg[m][ds] = {}
            for met in METRICS:
                vals = [results.get((ds, seed, m), {}).get(met) if results.get((ds, seed, m)) else None
                        for seed in SEEDS]
                agg[m][ds][met] = mean_std(vals)

    # ---- 6. 写对比 CSV + 终端打印 ----
    print("[汇总] 写出对比表 ...")
    write_compare_csv(out_dir / "summary_compare.csv", agg)
    print_compare_table(agg)

    print(f"\n[完成] CSV 已输出到: {out_dir}")
    print("       summary_hq.csv / summary_full.csv / summary_compare.csv")


if __name__ == "__main__":
    main()
