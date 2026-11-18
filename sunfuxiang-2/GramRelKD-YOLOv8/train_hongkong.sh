#!/usr/bin/env bash
#
# Train the 4 Hong Kong wall-defect network architectures with default settings,
# then print each architecture's test-set metrics to the terminal.
#
# Launcher : main.py   (mode selected by flags: training by default, --val for test eval)
# Settings : ultralytics/cfg/default.yaml  (epochs=300, batch=32, imgsz=640, split=test, ...)
# Dataset  : ultralytics/cfg/datasets/coco8.yaml  ->  Hong Kong wall-defect (spalling/crack/moisture)
#
# Usage    : bash train_hongkong.sh
#
set -euo pipefail

REPO=/data/cle/sunfuxiang-2/ultralytics-main
cd "$REPO"

# "name:model_yaml" pairs. The name both labels the run and locates best.pt in phase 2.
MODELS=(
  "L1-B:ultralytics/cfg/models/hongkong/L1-B.yaml"
  "L1-N:ultralytics/cfg/models/hongkong/L1-N.yaml"
  "L1-96:ultralytics/cfg/models/hongkong/L1-96.yaml"
  "C2f_SAPA:ultralytics/cfg/models/hongkong/C2f_SAPA.yaml"
)

echo "########################################################"
echo "# Phase 1/2 : TRAINING (default settings, split=test)  #"
echo "########################################################"
for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"
  yaml="${entry##*:}"
  echo
  echo "======== TRAIN  ${name}  (${yaml}) ========"
  python main.py --model "${yaml}" --name "hongkong_${name}"
done

echo
echo "########################################################"
echo "# Phase 2/2 : TEST-SET RESULTS PER ARCHITECTURE        #"
echo "########################################################"
for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"
  weights="runs/detect/hongkong_${name}/weights/best.pt"
  echo
  echo "======== TEST   ${name}  (${weights}) ========"
  if [[ -f "${weights}" ]]; then
    python main.py --val "${weights}"
  else
    echo "[WARN] weights not found: ${weights} — skipping"
  fi
done

echo
echo "All done."
