#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="/data/cle/sunfuxiang-1/ours"
DATASET_ROOT="/data/cle/sunfuxiang-1/dataset"
PYTHON_BIN="${PYTHON_BIN:-/home/login/miniconda3/envs/yolo/bin/python}"
DEFAULT_CFG="${REPO_DIR}/ultralytics/cfg/default.yaml"
OUTPUT_PROJECT="${REPO_DIR}/runs/main_all_datasets"
LOG_DIR="${REPO_DIR}/train_logs/main_all_datasets_sequential"

if [[ ! -f "${REPO_DIR}/main.py" ]]; then
    echo "ERROR: main.py not found: ${REPO_DIR}/main.py" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${DEFAULT_CFG}" ]]; then
    echo "ERROR: default config not found: ${DEFAULT_CFG}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_PROJECT}" "${LOG_DIR}"

WORK_DIR="$(mktemp -d "${REPO_DIR}/.main_all_datasets.XXXXXX")"
cleanup() {
    case "${WORK_DIR}" in
        "${REPO_DIR}"/.main_all_datasets.*)
            rm -f -- "${WORK_DIR}/bootstrap/sitecustomize.py"
            rm -rf -- "${WORK_DIR}/run" "${WORK_DIR}/datasets"
            rmdir -- "${WORK_DIR}/bootstrap" "${WORK_DIR}" 2>/dev/null || true
            ;;
        *)
            echo "WARNING: refusing to clean unexpected path: ${WORK_DIR}" >&2
            ;;
    esac
}
trap cleanup EXIT INT TERM

mkdir -p \
    "${WORK_DIR}/bootstrap" \
    "${WORK_DIR}/run/ultralytics/cfg" \
    "${WORK_DIR}/datasets"

# main.py does not accept command-line data arguments. It relies on
# TASK2DATA["detect"] == "coco8.yaml". This startup hook changes that default
# to the absolute YAML selected for the current dataset without editing source.
cat > "${WORK_DIR}/bootstrap/sitecustomize.py" <<'PY'
import os
import ultralytics.cfg as cfg

data_yaml = os.environ.get("OURS_DATA_YAML")
if data_yaml:
    cfg.TASK2DATA["detect"] = data_yaml
PY

cat > "${WORK_DIR}/datasets/jptomato.yaml" <<YAML
train: ${DATASET_ROOT}/jptomato/train
val: ${DATASET_ROOT}/jptomato/val
test: ${DATASET_ROOT}/jptomato/test
nc: 6
names: ['i', 'a', 'e', 'is', 'q', 't']
YAML

cat > "${WORK_DIR}/datasets/raspberry.yaml" <<YAML
train: ${DATASET_ROOT}/raspberry/train
val: ${DATASET_ROOT}/raspberry/val
test: ${DATASET_ROOT}/raspberry/test
nc: 5
names: ['is', 'a', 'e', 'i', 'yt']
YAML

cat > "${WORK_DIR}/datasets/strawberry.yaml" <<YAML
train: ${DATASET_ROOT}/strawberry/train
val: ${DATASET_ROOT}/strawberry/val
test: ${DATASET_ROOT}/strawberry/test
nc: 3
names: ['i', 'a', 'e']
YAML

cat > "${WORK_DIR}/datasets/tomato.yaml" <<YAML
train: ${DATASET_ROOT}/tomato/train
val: ${DATASET_ROOT}/tomato/val
test: ${DATASET_ROOT}/tomato/test
nc: 2
names: ['unripe', 'ripe']
YAML

cat > "${WORK_DIR}/datasets/VisDrone2019.yaml" <<YAML
path: ${DATASET_ROOT}/VisDrone2019
train: images/train
val: images/val
test: images/test
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
YAML

declare -A RUN_NAMES
declare -A BEST_WEIGHTS

train_one_dataset() {
    local dataset_name="$1"
    local data_yaml="${WORK_DIR}/datasets/${dataset_name}.yaml"
    local run_stamp
    local run_name
    local run_dir
    local best_weight
    local train_log

    run_stamp="$(date '+%Y%m%d_%H%M%S')"
    run_name="mainpy_${dataset_name}_${run_stamp}"
    run_dir="${OUTPUT_PROJECT}/${run_name}"
    best_weight="${run_dir}/weights/best.pt"
    train_log="${LOG_DIR}/${run_name}_train.log"

    if [[ ! -d "${DATASET_ROOT}/${dataset_name}" ]]; then
        echo "ERROR: dataset directory not found: ${DATASET_ROOT}/${dataset_name}" >&2
        return 1
    fi

    # main.py explicitly loads ultralytics/cfg/default.yaml relative to cwd.
    # Supply an isolated copy so the repository's default.yaml is untouched.
    sed \
        -e "s|^project:.*|project: ${OUTPUT_PROJECT}|" \
        -e "s|^name:.*|name: ${run_name}|" \
        -e "s|^split:.*|split: val|" \
        "${DEFAULT_CFG}" > "${WORK_DIR}/run/ultralytics/cfg/default.yaml"

    echo
    echo "================================================================"
    echo "TRAIN: ${dataset_name}"
    echo "data:  ${data_yaml}"
    echo "run:   ${run_dir}"
    echo "================================================================"

    (
        cd "${WORK_DIR}/run"
        OURS_DATA_YAML="${data_yaml}" \
        PYTHONPATH="${WORK_DIR}/bootstrap:${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" "${REPO_DIR}/main.py"
    ) 2>&1 | tee "${train_log}"

    if [[ ! -f "${best_weight}" ]]; then
        echo "ERROR: best.pt not found after training: ${best_weight}" >&2
        return 1
    fi

    RUN_NAMES["${dataset_name}"]="${run_name}"
    BEST_WEIGHTS["${dataset_name}"]="${best_weight}"
    echo "TRAINING FINISHED: ${dataset_name}"
    echo "best weights: ${best_weight}"
}

test_one_dataset() {
    local dataset_name="$1"
    local data_yaml="${WORK_DIR}/datasets/${dataset_name}.yaml"
    local run_name="${RUN_NAMES[${dataset_name}]}"
    local best_weight="${BEST_WEIGHTS[${dataset_name}]}"
    local test_log="${LOG_DIR}/${run_name}_test.log"

    echo
    echo "================================================================"
    echo "TEST RESULT: ${dataset_name}"
    echo "weights: ${best_weight}"
    echo "================================================================"

    (
        cd "${REPO_DIR}"
        "${PYTHON_BIN}" - \
            "${best_weight}" \
            "${data_yaml}" \
            "${OUTPUT_PROJECT}" \
            "${run_name}_test" <<'PY'
import sys
from ultralytics import YOLO

best_weight, data_yaml, project, test_name = sys.argv[1:5]
YOLO(best_weight).val(
    data=data_yaml,
    split="test",
    imgsz=640,
    batch=32,
    device=0,
    workers=8,
    project=project,
    name=test_name,
    exist_ok=False,
)
PY
    ) 2>&1 | tee "${test_log}"

    echo "FINISHED: ${dataset_name}"
}

DATASET_ORDER=(
    jptomato
    raspberry
    strawberry
    tomato
    VisDrone2019
)

echo
echo "================================================================"
echo "PHASE 1/2: TRAIN ALL DATASETS"
echo "================================================================"
for dataset_name in "${DATASET_ORDER[@]}"; do
    train_one_dataset "${dataset_name}"
done

echo
echo "================================================================"
echo "ALL TRAINING FINISHED"
echo "PHASE 2/2: TEST ALL DATASETS"
echo "================================================================"
for dataset_name in "${DATASET_ORDER[@]}"; do
    test_one_dataset "${dataset_name}"
done

echo
echo "================================================================"
echo "ALL TRAINING AND TESTING FINISHED"
echo "runs: ${OUTPUT_PROJECT}"
echo "logs: ${LOG_DIR}"
echo "================================================================"
