#!/usr/bin/env bash
set -uo pipefail

project=/data/cle/sunfuxiang-2/ultralytics-main
python=/home/login/miniconda3/envs/yolo/bin/python
run_dir="$project/runs/detect/hongkong_SR1_P4Only_MTRD"
report_dir="$project/runs/analysis/mtrd_formal_result"
implementation_launch="$project/runs/analysis/mtrd_implementation_check/train_launch_results.json"
formal_launch="$report_dir/formal_train_launch_results.json"
log="$report_dir/formal_mtrd_train.log"

cd "$project" || exit 90

if [ -e "$run_dir/results.csv" ] || [ -e "$run_dir/weights/best.pt" ] || [ -e "$run_dir/weights/last.pt" ]; then
  echo "FAILED: formal run directory already contains results: $run_dir" >&2
  exit 91
fi
if [ -e "$implementation_launch" ]; then
  echo "FAILED: implementation train_launch_results.json unexpectedly exists; refusing to overwrite it" >&2
  exit 92
fi

mkdir -p "$report_dir"
start_iso=$(date -Is)
start_epoch=$(date +%s)
teacher_hash_before=$(sha256sum runs/detect/runs/detect/hongkong_L1-96/weights/best.pt | awk '{print $1}')
student_yaml_hash=$(sha256sum ultralytics/cfg/models/hongkong/yolov8n-L1-96-SR1.yaml | awk '{print $1}')
ultralytics_version=$($python -c 'import ultralytics; print(ultralytics.__version__)')

{
  echo "FORMAL_MTRD_START=$start_iso"
  echo "EXPECTED_SAVE_DIR=$run_dir"
  echo "TEACHER_HASH_BEFORE=$teacher_hash_before"
  echo "STUDENT_YAML_HASH=$student_yaml_hash"
  echo "ULTRALYTICS_VERSION=$ultralytics_version"
  echo "COMMAND=$python scripts/markov_distill_train.py --mode train --temperature 0.05 --transition-steps 1,2 --step-weights 1.0,1.0 --self-loop-prob 0.0 --kd-weight 0.10246292566479759 --name hongkong_SR1_P4Only_MTRD"
} | tee "$log"

$python scripts/markov_distill_train.py \
  --mode train \
  --temperature 0.05 \
  --transition-steps 1,2 \
  --step-weights 1.0,1.0 \
  --self-loop-prob 0.0 \
  --kd-weight 0.10246292566479759 \
  --name hongkong_SR1_P4Only_MTRD \
  2>&1 | tee -a "$log"
train_status=${PIPESTATUS[0]}

end_iso=$(date -Is)
end_epoch=$(date +%s)
elapsed_seconds=$((end_epoch - start_epoch))
teacher_hash_after=$(sha256sum runs/detect/runs/detect/hongkong_L1-96/weights/best.pt | awk '{print $1}')

if [ -e "$implementation_launch" ]; then
  mv "$implementation_launch" "$formal_launch"
fi

{
  echo "FORMAL_MTRD_END=$end_iso"
  echo "TRAIN_EXIT_STATUS=$train_status"
  echo "ELAPSED_SECONDS=$elapsed_seconds"
  echo "TEACHER_HASH_AFTER=$teacher_hash_after"
  if [ "$teacher_hash_before" = "$teacher_hash_after" ]; then
    echo "TEACHER_HASH_UNCHANGED=true"
  else
    echo "TEACHER_HASH_UNCHANGED=false"
  fi
} | tee -a "$log" | tee "$report_dir/formal_train_status.txt"

exit "$train_status"
