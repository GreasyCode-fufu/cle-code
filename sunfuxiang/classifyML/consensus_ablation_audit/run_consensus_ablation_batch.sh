#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON=/home/login/miniconda3/envs/sunfuxiang/bin/python
REPO=/home/login/data/cle/sunfuxiang/classifyML
DATA_ROOT=/home/login/data/cle/sunfuxiang/datasets
MANIFEST_DIR="$REPO/manifests/consensus_ablation"
OUT_ROOT="$REPO/weights_consensus_ablation"
PRETRAINED="$REPO/pth/resnet34-333f7ec4.pth"
MASTER_LOG="$OUT_ROOT/master_training.log"
VERIFY="$REPO/scripts/verify_consensus_run.py"

mkdir -p "$OUT_ROOT"
touch "$MASTER_LOG"

current_subset="startup"
current_seed="-"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$MASTER_LOG"
}

fail() {
  local message="$1"
  log "FAILED subset=$current_subset seed=$current_seed reason=$message"
  exit 1
}

on_error() {
  local rc=$?
  local line=${BASH_LINENO[0]:-unknown}
  log "FAILED subset=$current_subset seed=$current_seed reason=unexpected_exit rc=$rc line=$line"
  exit "$rc"
}
trap on_error ERR

check_gpu() {
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader >> "$MASTER_LOG" 2>&1 || fail "nvidia-smi_unavailable"
}

verify_manifest() {
  local subset=$1
  "$PYTHON" - "$MANIFEST_DIR/MANIFEST_METADATA.json" "$MANIFEST_DIR/$subset.csv" "$subset" <<'PY'
import hashlib, json, pathlib, sys
metadata_path = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
subset = sys.argv[3]
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
expected = metadata["subsets"][subset]["sha256"]
if actual != expected:
    raise SystemExit(f"manifest hash mismatch for {subset}: {actual} != {expected}")
print(f"manifest {subset} sha256={actual}")
PY
}

train_complete() {
  "$PYTHON" "$VERIFY" --repo "$REPO" --subset "$1" --seed "$2" >/dev/null 2>&1
}

run_complete() {
  "$PYTHON" "$VERIFY" --repo "$REPO" --subset "$1" --seed "$2" --require-eval >/dev/null 2>&1
}

evaluate_run() {
  local subset=$1 seed=$2 model_dir=$3 weights=$4 eval_dir=$5 eval_log=$6
  if [[ -d "$eval_dir" ]] && ! run_complete "$subset" "$seed"; then
    local backup="${eval_dir}.failed.$(date +%Y%m%d_%H%M%S)"
    mv "$eval_dir" "$backup"
    log "EVAL_RETRY_PRESERVED subset=$subset seed=$seed path=$backup"
  fi
  log "EVAL_START subset=$subset seed=$seed"
  if ! "$PYTHON" "$REPO/evaluate.py" \
      --model resnet34 \
      --weights "$weights" \
      --data-path "$DATA_ROOT/3class/test" \
      --class-names Benign_lesions Malignant_lesions body_of_stomach_normal \
      --img-size 224 \
      --batch-size 64 \
      --device cuda:0 \
      --out-dir "$eval_dir" > "$eval_log" 2>&1; then
    fail "evaluation_exit_nonzero log=$eval_log"
  fi
  if ! run_complete "$subset" "$seed"; then
    fail "post_evaluation_validation_failed log=$eval_log"
  fi
  log "EVAL_DONE subset=$subset seed=$seed predictions=$eval_dir/predictions.csv"
}

log "BATCH_START order=R,V,C,RV,RC,VC seeds=42,43,44"

for subset in R V C RV RC VC; do
  for seed in 42 43 44; do
    current_subset=$subset
    current_seed=$seed
    run_root="$OUT_ROOT/$subset/seed$seed"
    model_dir="$run_root/resnet34"
    best="$model_dir/resnet34_${subset}_seed${seed}_best.pth"
    train_log="$model_dir/train_log_resnet34_${subset}_seed${seed}.csv"
    stdout_log="$run_root/train_stdout.log"
    eval_log="$run_root/evaluation.log"
    eval_dir="$model_dir/eval_fixed_test"

    check_gpu
    if ! verify_manifest "$subset" >> "$MASTER_LOG" 2>&1; then
      fail "manifest_hash_mismatch"
    fi

    if run_complete "$subset" "$seed"; then
      log "SKIP_DONE subset=$subset seed=$seed reason=complete_validated"
      continue
    fi

    if train_complete "$subset" "$seed"; then
      log "TRAIN_DONE subset=$subset seed=$seed reason=existing_train_validated"
      evaluate_run "$subset" "$seed" "$model_dir" "$best" "$eval_dir" "$eval_log"
      continue
    fi

    if [[ -d "$run_root" ]] && find "$run_root" -mindepth 1 -print -quit | grep -q .; then
      fail "incomplete_existing_run_preserved path=$run_root"
    fi

    mkdir -p "$run_root"
    log "START subset=$subset seed=$seed timestamp=$(date --iso-8601=seconds)"
    if ! "$PYTHON" "$REPO/train_cle_fair.py" \
        --model resnet34 \
        --num_classes 3 \
        --img_size 224 \
        --batch-size 32 \
        --epochs 300 \
        --lr 0.0001 \
        --label-smoothing 0 \
        --seed "$seed" \
        --device cuda:0 \
        --early-stop 30 \
        --weights "$PRETRAINED" \
        --data-path-train "$DATA_ROOT/3class/train" \
        --train-manifest "$MANIFEST_DIR/$subset.csv" \
        --data-path-val "$DATA_ROOT/3class/val" \
        --save-dir "$run_root" \
        --save-name "resnet34_${subset}_seed${seed}_best.pth" \
        --csv-name "train_log_resnet34_${subset}_seed${seed}.csv" \
        --norm-cache-dir "$REPO/norm_cache" > "$stdout_log" 2>&1; then
      fail "training_exit_nonzero log=$stdout_log"
    fi
    if ! train_complete "$subset" "$seed"; then
      fail "post_training_validation_failed log=$stdout_log"
    fi
    log "TRAIN_DONE subset=$subset seed=$seed checkpoint=$best train_log=$train_log"
    evaluate_run "$subset" "$seed" "$model_dir" "$best" "$eval_dir" "$eval_log"
  done
done

current_subset="summary"
current_seed="-"
log "SUMMARY_START"
"$PYTHON" "$REPO/scripts/summarize_consensus_ablation.py" \
  --legacy-map "$REPO/consensus_ablation_audit/LEGACY_PREDICTION_MAP.json" \
  --new-root "$OUT_ROOT" \
  --out-dir "$REPO/consensus_ablation_results" \
  > "$OUT_ROOT/summary_stdout.log" 2>&1
"$PYTHON" "$REPO/scripts/finalize_consensus_ablation.py" \
  --metrics "$REPO/consensus_ablation_results/CONSENSUS_ABLATION_METRICS.json" \
  --out-dir "$REPO/consensus_ablation_results" \
  >> "$OUT_ROOT/summary_stdout.log" 2>&1
log "SUMMARY_DONE output=$REPO/consensus_ablation_results"
log "BATCH_DONE"
