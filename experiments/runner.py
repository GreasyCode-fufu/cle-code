#!/usr/bin/env python3
"""Classification 自动化实验调度器
用法: nohup python experiments/runner.py experiments/jobs_batchX.json 3 >> experiments/batchX.nohup.log 2>&1 &
  - jobs 文件: [{"name": ..., "model": ..., "weights": ..., "seed": ...}, ...]
  - 第2参数: 并发数(默认3)
  - 每个任务: 训练(300ep,早停30) -> val评估 -> test评估 -> 追加 experiments/results.csv
  - 单任务失败不影响其他任务(status=error 记录)
"""
import csv, glob, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

REPO = "/home/login/data/cle/sunfuxiang/Classification"
PY = "/home/login/miniconda3/envs/sunfuxiang/bin/python"
PTH = "/home/login/data/cle/sunfuxiang/classifyML/pth"
RESULTS = os.path.join(REPO, "experiments", "results.csv")
LOGDIR = os.path.join(REPO, "experiments", "logs")

FIELDS = ["ts", "batch", "model", "seed", "status", "val_f1", "test_f1", "test_acc",
          "test_auroc", "sens_benign", "sens_malignant", "sens_normal", "elapsed_min", "note"]


def sh(cmd, log_path):
    with open(log_path, "a") as lf:
        lf.write(f"\n===== [{datetime.now()}] {cmd}\n")
        p = subprocess.run(cmd, shell=True, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT)
    return p.returncode


def latest_eval_dir(model):
    dirs = sorted(glob.glob(os.path.join(REPO, "weights", model, "eval_*")))
    return dirs[-1] if dirs else None


def read_metric(model, which):
    """which: 'val'|'test' — evaluate 后从最近的 eval 目录读 results.json"""
    d = latest_eval_dir(model)
    if not d:
        return None
    r = json.load(open(os.path.join(d, "results.json")))
    per = r.get("per_class", {})
    def sens(key):
        m = per.get(key, {})
        return round(m.get("sensitivity", m.get("recall", float("nan"))), 4)
    return {
        "f1": round(r.get("macro_f1", float("nan")), 4),
        "acc": round(r.get("accuracy", float("nan")), 4),
        "auroc": round(r.get("macro_auroc", float("nan")), 4),
        "sb": sens("Benign_lesions"), "sm": sens("Malignant_lesions"),
        "sn": sens("body_of_stomach_normal"), "dir": os.path.basename(d),
    }


def append_row(row):
    newfile = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if newfile:
            w.writeheader()
        w.writerow(row)


def run_job(batch, job, worker_id):
    model, seed = job["model"], job.get("seed", 42)
    weights = job.get("weights") or os.path.join(PTH, job.get("pth", ""))
    name = job.get("name", model)
    log = os.path.join(LOGDIR, f"{name}_seed{seed}.log")
    t0 = time.time()
    row = {"ts": datetime.now().strftime("%m-%d %H:%M"), "batch": batch, "model": name,
           "seed": seed, "status": "error", "note": ""}
    try:
        rc = sh(f"{PY} train_cle_fair.py --model {model} --num_classes 3 --img_size 224 "
                f"--seed {seed} --data-path-train ./datasets/cle/train "
                f"--data-path-val ./datasets/cle/val --weights {weights} "
                f"--save-dir ./weights --epochs 300", log)
        if rc != 0:
            row["note"] = "train_failed"; raise RuntimeError("train rc=" + str(rc))
        best = os.path.join(REPO, "weights", model, "model_best.pth")
        # val 评估(选模参考)
        sh(f"{PY} evaluate.py --model {model} --weights {best} --data-path ./datasets/cle/val", log)
        v = read_metric(model, "val")
        # test 评估(最终判定)
        sh(f"{PY} evaluate.py --model {model} --weights {best} --data-path ./datasets/cle/test", log)
        t = read_metric(model, "test")
        if not v or not t:
            row["note"] = "eval_missing"; raise RuntimeError("eval parse failed")
        row.update({"status": "ok", "val_f1": v["f1"], "test_f1": t["f1"], "test_acc": t["acc"],
                    "test_auroc": t["auroc"], "sens_benign": t["sb"], "sens_malignant": t["sm"],
                    "sens_normal": t["sn"],
                    "elapsed_min": round((time.time() - t0) / 60, 1),
                    "note": t["dir"]})
    except Exception as e:
        row["elapsed_min"] = round((time.time() - t0) / 60, 1)
        row["note"] = f"{row['note']} {e}"[:120]
    append_row(row)
    print(f"[done] {name} seed{seed}: {row['status']} val={row.get('val_f1')} test={row.get('test_f1')} ({row['elapsed_min']}min)", flush=True)


def main():
    jobs_file = sys.argv[1]
    conc = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    batch = os.path.basename(jobs_file).replace("jobs_", "").replace(".json", "")
    jobs = json.load(open(jobs_file))
    os.makedirs(LOGDIR, exist_ok=True)
    print(f"[runner] batch={batch} jobs={len(jobs)} conc={conc} start={datetime.now()}", flush=True)
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(run_job, batch, j, i) for i, j in enumerate(jobs)]
        for f in as_completed(futs):
            f.result()
    print(f"[runner] batch={batch} ALL DONE {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
