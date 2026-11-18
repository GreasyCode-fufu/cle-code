#!/bin/bash
# 消融链泛化补齐(除BUSI): 每域补 v2m(MSP-only) / v2c(+DCH) 缺口, tiny/v6f 已有
set -u
cd /home/login/data/cle/auto/ZCode/Classification
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
run_ds () { local name=$1 data=$2 ncls=$3 conc=$4; shift 4
  local jobs=experiments/jobs_abl_${name}.json
  python3 - "$jobs" "$@" <<'PYEOF'
import json, sys
out, specs = sys.argv[1], sys.argv[2:]
jobs = []
for sp in specs:                      # 格式 model:seed1,seed2
    m, seeds = sp.split(":")
    jobs += [{"model": m, "pth": "convnext_tiny_1k_224_ema.pth", "seed": int(s)}
             for s in seeds.split(",")]
json.dump(jobs, open(out, "w"), indent=1)
PYEOF
  local classes; classes=$(ls "$data/train" | sort | tr '\n' ' ')
  echo "[abl:$name] $(($# - 4)) jobs"
  RUN_DATA="$data" RUN_WEIGHTS="weights_xd_${name}_abl" \
  RUN_RESULTS="experiments/results_xd_${name}_abl.csv" RUN_NCLS=$ncls RUN_CLASSES="$classes" \
    $PY experiments/runner.py "$jobs" "$conc" >> experiments/abl_${name}.nohup.log 2>&1
}
run_ds heel  datasets/heelBone 3 4 "convnext_v2m:42,43,44"
run_ds covid datasets/covid_v1 3 4 "convnext_v2m:42,43,44"
run_ds cub   datasets/cub_v1 200 3 "convnext_v2m:42,43,44" "convnext_v2c:43,44"
run_ds tcga  datasets/TCGA    6 3 "convnext_v2m:42,43,44" "convnext_v2c:42,43,44"
echo "[abl] ALL DONE $(date)"
