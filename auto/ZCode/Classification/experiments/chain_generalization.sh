#!/bin/bash
# R3 泛化波次: 每数据集 [final_model + convnext_tiny] x seeds 42/43/44, 同批同协议
# 类名自动从数据集 train 目录推导(evaluate.py 需要与目录序一致的 --class-names)
set -u
cd /home/login/data/cle/auto/ZCode/Classification
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
FINAL="${FINAL:?need FINAL model name}"
R3SUFFIX="r3"

run_ds () {  # name data ncls conc
  local name=$1 data=$2 ncls=$3 conc=$4
  local jobs=experiments/jobs_gen_${name}_${R3SUFFIX}.json
  python3 - "$FINAL" "$jobs" <<'PYEOF'
import json, sys
final, out = sys.argv[1], sys.argv[2]
jobs = [{"model": m, "pth": "convnext_tiny_1k_224_ema.pth", "seed": s}
        for m in [final, "convnext_tiny"] for s in (42, 43, 44)]
json.dump(jobs, open(out, "w"), indent=1)
PYEOF
  local classes
  classes=$(ls "$data/train" | sort | tr '\n' ' ')
  echo "[gen:$name] classes= $classes"
  RUN_DATA="$data" RUN_WEIGHTS="weights_xd_${name}_${R3SUFFIX}" \
  RUN_RESULTS="experiments/results_xd_${name}_${R3SUFFIX}.csv" RUN_NCLS=$ncls \
  RUN_CLASSES="$classes" \
    $PY experiments/runner.py "$jobs" "$conc" >> experiments/gen_${name}_${R3SUFFIX}.nohup.log 2>&1
}

run_ds busi   datasets/busi_v1                       3 4
run_ds heel   datasets/heelBone                      3 4
run_ds covid  datasets/covid_v1                      3 4
run_ds cub    datasets/cub_v1                      200 3
run_ds tcga   /home/login/data/cle/auto/datasets/TCGA 6 3
echo "[gen] ALL DONE $(date)"
