#!/usr/bin/env bash
set -euo pipefail

root=/data/cle/sunfuxiang-2/GramRelKD-YOLOv8
python=/home/login/miniconda3/envs/yolo/bin/python
name=hongkong_SR1_P4Only_RelKD_clone_reproduction

if [[ "${1:-}" != "--train" ]]; then
  printf '%s\n' "Safety stop: this launcher does not train unless the first argument is --train."
  printf '%s\n' "FINAL MAIN METHOD: GRAM RELKD"
  exit 2
fi

if [[ -e "$root/runs/detect/$name" ]]; then
  printf 'Refusing to overwrite existing run: %s\n' "$root/runs/detect/$name" >&2
  exit 3
fi

cd "$root"
exec "$python" scripts/distill_train_clone.py \
  --mode train \
  --teacher "$root/runs/detect/runs/detect/hongkong_L1-96/weights/best.pt" \
  --student-yaml "$root/ultralytics/cfg/models/hongkong/yolov8n-L1-96-SR1.yaml" \
  --data "$root/ultralytics/cfg/datasets/coco8.yaml" \
  --cfg "$root/ultralytics/cfg/default.yaml" \
  --project "$root/runs/detect" \
  --name "$name" \
  --epochs 300 \
  --batch 32 \
  --imgsz 640 \
  --workers 8 \
  --device 0 \
  --seed 0 \
  --kd-weight 0.0431658384 \
  --kd-warmup-epochs 10 \
  --kd-pool-size 20 \
  --kd-layer-weights 1.0,1.0,1.0

