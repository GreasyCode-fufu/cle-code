#!/bin/bash
# R6-B 复检批: CLE-Net 项目环境(torch 2.5.1) 重跑 CLE-Net/基线/4 对比网络, 3 seeds 42/43/44。
# 权重目录 weights_r6b_recheck/(全新, 不触碰 Codex 任何已有结果); 结果 CSV 写本目录。
set -u
cd "$(dirname "$0")"
L=$(dirname "$(pwd)")                       # prueNetwork/CLE-Net
PY=/home/login/miniconda3/envs/cle_topnets/bin/python
DATA=/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1
PTH=/home/login/data/cle/sunfuxiang/classifyML/pth
WROOT=$L/weights_r6b_recheck
mkdir -p logs $WROOT
run () { local model=$1 pre=$2 seed=$3
  local sd=$WROOT/${model}_seed${seed}
  rm -rf "$sd"
  echo "[train] $model seed$seed $(date +%H:%M)"
  $PY run_one.py --model $model --num_classes 3 --img_size 224 --seed $seed \
    --data-path-train $DATA/train --data-path-val $DATA/val \
    --weights "$PTH/$pre" --save-dir "$sd" --epochs 300 --early-stop 30 \
    > logs/${model}_seed${seed}.log 2>&1
  local rc=$?
  [ $rc -ne 0 ] && { echo "[FAIL-train] $model seed$seed rc=$rc"; return; }
  $PY $L/evaluate.py --model $model --weights $sd/$model/model_best.pth \
    --data-path $DATA/test --out-dir $sd/$model/eval_test > /dev/null 2>&1 \
    && echo "[done] $model seed$seed $(date +%H:%M)" || echo "[FAIL-eval] $model seed$seed"
}
export -f run; export PY DATA PTH WROOT L
for spec in "jumbo_small sota2026/jumbo_small.pth" "gmnet_s3 sota2026/gmnet_s3.pth" \
            "iformer_s sota2026/iformer_s.pth" "lsnet_s sota2026/lsnet_s.pth"; do
  set -- $spec
  for s in 42 43 44; do echo "$1 $2 $s"; done
done | xargs -P 3 -L 1 bash -c 'run "$@"' _
echo "[R6-B recheck batch ALL DONE $(date)]"
