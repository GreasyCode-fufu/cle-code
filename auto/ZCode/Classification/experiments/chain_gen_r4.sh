#!/bin/bash
# R4 泛化: v2c(最终网络) 各域 3seeds; busi/heel/covid/cub 复用 r3 批同日同管线 tiny 基线
set -u
cd /home/login/data/cle/auto/ZCode/Classification
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
run_ds () {  # name data ncls conc models...
  local name=$1 data=$2 ncls=$3 conc=$4; shift 4
  local jobs=experiments/jobs_gen4_${name}.json
  python3 - "$jobs" "$@" <<'PYEOF'
import json, sys
out, models = sys.argv[1], sys.argv[2:]
jobs = [{"model": m, "pth": "convnext_tiny_1k_224_ema.pth", "seed": s}
        for m in models for s in (42, 43, 44)]
json.dump(jobs, open(out, "w"), indent=1)
PYEOF
  local classes; classes=$(ls "$data/train" | sort | tr '\n' ' ')
  echo "[gen4:$name] classes= $classes"
  RUN_DATA="$data" RUN_WEIGHTS="weights_xd_${name}_r4" \
  RUN_RESULTS="experiments/results_xd_${name}_r4.csv" RUN_NCLS=$ncls RUN_CLASSES="$classes" \
    $PY experiments/runner.py "$jobs" "$conc" >> experiments/gen4_${name}.nohup.log 2>&1
}
run_ds busi  datasets/busi_v1                       3 4 convnext_v2c
run_ds heel  datasets/heelBone                      3 4 convnext_v2c
run_ds covid datasets/covid_v1                      3 4 convnext_v2c
run_ds cub   datasets/cub_v1                      200 3 convnext_v2c
run_ds tcga  datasets/TCGA                          6 3 convnext_v2c convnext_tiny
echo "[gen4] ALL DONE $(date)"
