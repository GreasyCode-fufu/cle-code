#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/login/data/cle/auto/prueNetwork/CLE-Net
ENV=/home/login/miniconda3/envs/cle_topnets/bin/python
DATA=/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1
WEIGHTS=/home/login/data/cle/sunfuxiang/classifyML/pth/convnext_tiny_1k_224_ema.pth
OUT=$ROOT/cle_net_rerun_codex_20261109_v2
mkdir -p "$OUT/logs" "$OUT/results"
printf 'started_at=%s\n' "$(date -Is)" > "$OUT/LAUNCH_MANIFEST.txt"
printf 'host=%s\n' "$(hostname)" >> "$OUT/LAUNCH_MANIFEST.txt"
printf 'owner=%s\n' "$(id -un)" >> "$OUT/LAUNCH_MANIFEST.txt"

run_model() {
  local model="$1"; local seed="$2"; local safe="$3"
  local runbase="$OUT/results/${safe}_seed${seed}"
  mkdir -p "$runbase"
  echo "$(date -Is) START model=$model seed=$seed pid=$$" | tee -a "$OUT/LAUNCH_MANIFEST.txt"
  "$ENV" "$ROOT/latest_top_networks/run_train.py" \
    --model "$model" --num_classes 3 --img_size 224 --seed "$seed" \
    --data-path-train "$DATA/train" --data-path-val "$DATA/val" \
    --weights "$WEIGHTS" --save-dir "$runbase" \
    --epochs 300 --early-stop 30 --batch-size 32 --lr 1e-4 --device cuda:0 \
    > "$OUT/logs/${safe}_seed${seed}.train.log" 2>&1
  local checkpoint
  checkpoint="$(find "$runbase" -type f -name model_best.pth -print | sort | tail -n 1)"
  if [ -z "$checkpoint" ]; then echo "missing checkpoint for $model seed $seed" >&2; exit 2; fi
  local run="$(dirname "$checkpoint")"
  "$ENV" "$ROOT/latest_top_networks/run_eval.py" \
    --model "$model" --weights "$checkpoint" \
    --data-path "$DATA/test" --img-size 224 --run-meta "$run/run_meta.json" \
    --out-dir "$run/eval_test" --batch-size 64 --device cuda:0 \
    > "$OUT/logs/${safe}_seed${seed}.eval.log" 2>&1
  echo "$(date -Is) DONE model=$model seed=$seed" | tee -a "$OUT/LAUNCH_MANIFEST.txt"
}

# Sequential execution keeps one GPU slot free for Zcode and avoids cross-run interference.
for seed in 42 43 44; do run_model convnext_egb_scg "$seed" convnext_egb_scg; done
for seed in 42 43 44; do run_model convnext_tiny "$seed" convnext_tiny; done
printf 'finished_at=%s\n' "$(date -Is)" >> "$OUT/LAUNCH_MANIFEST.txt"
