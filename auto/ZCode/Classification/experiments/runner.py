#!/usr/bin/env python3
"""Classification 自动化实验调度器 v2
用法: nohup python experiments/runner.py experiments/jobs_batchX.json 4 >> experiments/batchX.nohup.log 2>&1 &
  - 每任务: 训练 -> val评估 -> test评估 -> 追加 experiments/results.csv
  - 权重与评估按 weights/<model>_seed<seed>/<model>/ 分目录(多种子不互相覆盖)
  - val/test 用 results.json 的 test_dir 字段判别(不依赖时间顺序)
  - 单任务失败隔离
"""
import csv, glob, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 自动探测所在仓库
PY = "/home/login/miniconda3/envs/sunfuxiang/bin/python"
PTH = "/home/login/data/cle/sunfuxiang/classifyML/pth"
# 2026-10-25 起数据集切换为 cle_ac_v1(旧 results.csv/weights 均为旧数据集时代, 不可比):
# 权重与结果落到独立目录, 防止读到旧数据集训练的陈旧权重
# 可用环境变量切换数据集(如 heelBone): RUN_DATA/RUN_WEIGHTS/RUN_RESULTS/RUN_CLASSES
DATA = os.environ.get("RUN_DATA", "datasets/cle_ac_v1")
WEIGHTS_ROOT = os.environ.get("RUN_WEIGHTS", "weights_cle_ac_v1")
RESULTS = os.path.join(REPO, os.environ.get("RUN_RESULTS", "experiments/results_cle_ac_v1.csv"))
LOGDIR = os.path.join(REPO, "experiments", "logs")
# 评估类名(空格分隔, 须与训练 sorted 目录序一致); 留空则用 evaluate.py 默认(CLE三类)
EVAL_CLASSES = os.environ.get("RUN_CLASSES", "").split()

FIELDS = ["ts", "batch", "model", "seed", "status", "val_f1", "test_f1", "test_acc",
          "test_auroc", "sens_benign", "sens_malignant", "sens_normal", "elapsed_min", "note"]


def sh(cmd, log_path):
    with open(log_path, "a") as lf:
        lf.write(f"\n===== [{datetime.now()}] {cmd}\n")
        cmd = "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 " + cmd
        return subprocess.run(cmd, shell=True, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT).returncode


def run_dir(model, seed):
    return os.path.join(REPO, WEIGHTS_ROOT, f"{model}_seed{seed}", model)


def read_metric(model, seed, which):
    """按 test_dir 字段找对应数据集的 eval results.json。"""
    for f in sorted(glob.glob(os.path.join(run_dir(model, seed), "eval_*", "results.json"))):
        r = json.load(open(f))
        if ("val" in r.get("test_dir", "")) == (which == "val"):
            per = r.get("per_class", {})
            def sen(k):
                return round(per.get(k, {}).get("sensitivity", float("nan")), 4)
            return {
                "f1": round(r["macro_f1"], 4),
                "acc": round(r["accuracy"], 4),
                "auroc": round(r.get("auc_roc_macro", float("nan")), 4),
                "sb": sen("Benign_lesions"), "sm": sen("Malignant_lesions"),
                "sn": sen("body_of_stomach_normal"),
            }
    return None


def append_row(row):
    newfile = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if newfile:
            w.writeheader()
        w.writerow(row)


def run_job(batch, job):
    model, seed = job["model"], job.get("seed", 42)
    weights = job.get("weights") or os.path.join(PTH, job.get("pth", ""))
    name = job.get("name", model)
    log = os.path.join(LOGDIR, f"{name}_seed{seed}.log")
    t0 = time.time()
    row = {"ts": datetime.now().strftime("%m-%d %H:%M"), "batch": batch, "model": name,
           "seed": seed, "status": "error", "note": ""}
    try:
        import shutil
        shutil.rmtree(os.path.join(REPO, WEIGHTS_ROOT, f"{model}_seed{seed}"),
                      ignore_errors=True)  # 防陈旧空目录触发自动后缀
        epochs = os.environ.get("RUN_EPOCHS", "300")
        early = os.environ.get("RUN_EARLY", "30")
        ncls = os.environ.get("RUN_NCLS", "3")
        extra = job.get("extra", "")
        rc = sh(f"{PY} train_cle_fair.py --model {model} --num_classes {ncls} --img_size 224 "
                f"--seed {seed} --data-path-train ./{DATA}/train "
                f"--data-path-val ./{DATA}/val --weights {weights} "
                f"--save-dir ./{WEIGHTS_ROOT}/{model}_seed{seed} "
                f"--epochs {epochs} --early-stop {early} {extra}", log)
        if rc != 0:
            row["note"] = "train_failed"
            raise RuntimeError(f"train rc={rc}")
        best = os.path.join(run_dir(model, seed), "model_best.pth")
        if not os.path.exists(best):
            row["note"] = "ckpt_missing"
            raise RuntimeError("model_best.pth not found at " + best)
        cls_args = ("--class-names " + " ".join(EVAL_CLASSES)) if EVAL_CLASSES else ""
        sh(f"{PY} evaluate.py --model {model} --weights {best} --data-path ./{DATA}/val {cls_args}", log)
        sh(f"{PY} evaluate.py --model {model} --weights {best} --data-path ./{DATA}/test {cls_args}", log)
        v, t = read_metric(model, seed, "val"), read_metric(model, seed, "test")
        if not v or not t:
            row["note"] = "eval_missing"
            raise RuntimeError("eval parse failed")
        row.update({"status": "ok", "val_f1": v["f1"], "test_f1": t["f1"], "test_acc": t["acc"],
                    "test_auroc": t["auroc"], "sens_benign": t["sb"], "sens_malignant": t["sm"],
                    "sens_normal": t["sn"], "elapsed_min": round((time.time() - t0) / 60, 1),
                    "note": ""})
    except Exception as e:
        row["elapsed_min"] = round((time.time() - t0) / 60, 1)
        row["note"] = f"{row['note']} {e}"[:120]
    append_row(row)
    print(f"[done] {name} seed{seed}: {row['status']} val={row.get('val_f1')} "
          f"test={row.get('test_f1')} ({row['elapsed_min']}min)", flush=True)


def main():
    jobs_file = sys.argv[1]
    conc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    batch = os.path.basename(jobs_file).replace("jobs_", "").replace(".json", "")
    jobs = json.load(open(jobs_file))
    os.makedirs(LOGDIR, exist_ok=True)
    print(f"[runner] batch={batch} jobs={len(jobs)} conc={conc} start={datetime.now()}", flush=True)
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(run_job, batch, j) for j in jobs]
        for f in as_completed(futs):
            f.result()
    print(f"[runner] batch={batch} ALL DONE {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
