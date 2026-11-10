#!/bin/bash
# R6-B 跨域批: BUSI -> covid -> CUB, 6 模型 x 3 seeds x 3 域(54 runs), 与历史各批目录完全隔离
cd /home/login/data/cle/auto/ZCode/Classification
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
run_ds () { local dom=$1 data=$2 ncls=$3
  local classes; classes=$(ls "$data/train" | sort | tr '\n' ' ')
  RUN_DATA="$data" RUN_WEIGHTS="weights_xd_${dom}_r6b" \
  RUN_RESULTS="experiments/results_xd_${dom}_r6b.csv" RUN_NCLS=$ncls RUN_CLASSES="$classes" \
    $PY experiments/runner.py "experiments/jobs_xdr6b_${dom}.json" 3 >> "experiments/xdr6b_${dom}.nohup.log" 2>&1
  echo "[xdr6b:$dom] ALL DONE $(date)" >> experiments/xdr6b_chain.log
}
run_ds busi  datasets/busi_v1  3
run_ds covid datasets/covid_v1 3
run_ds cub   datasets/cub_v1   200
echo "[xdr6b] ALL DONE $(date)" >> experiments/xdr6b_chain.log
