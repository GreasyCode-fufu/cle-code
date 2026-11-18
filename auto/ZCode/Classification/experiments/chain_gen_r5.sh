#!/bin/bash
# R5 泛化: 最终网络 v6f(CH+FGS) 3seeds; 4小域复用同批tiny基线(r3批), TCGA跑双模型
set -u
cd /home/login/data/cle/auto/ZCode/Classification
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
run_ds () { local name=$1 data=$2 ncls=$3 conc=$4; shift 4
  local jobs=experiments/jobs_gen5_${name}.json
  python3 - "$jobs" "$@" <<'PYEOF'
import json, sys
out, models = sys.argv[1], sys.argv[2:]
json.dump([{"model": m, "pth": "convnext_tiny_1k_224_ema.pth", "seed": s}
           for m in models for s in (42, 43, 44)], open(out, "w"), indent=1)
PYEOF
  local classes; classes=$(ls "$data/train" | sort | tr '\n' ' ')
  echo "[gen5:$name] start"
  RUN_DATA="$data" RUN_WEIGHTS="weights_xd_${name}_r5" \
  RUN_RESULTS="experiments/results_xd_${name}_r5.csv" RUN_NCLS=$ncls RUN_CLASSES="$classes" \
    $PY experiments/runner.py "$jobs" "$conc" >> experiments/gen5_${name}.nohup.log 2>&1
}
run_ds busi  datasets/busi_v1  3 4 convnext_v6f
run_ds heel  datasets/heelBone 3 4 convnext_v6f
run_ds covid datasets/covid_v1 3 4 convnext_v6f
run_ds cub   datasets/cub_v1 200 3 convnext_v6f
run_ds tcga  datasets/TCGA    6 3 convnext_v6f convnext_tiny
echo "[gen5] ALL DONE $(date)"
