#!/usr/bin/env bash
set -euo pipefail

# Train EnergyGatedOCSPDConv scale=s using the best n-size trial params:
# runs/tune_energy_gated_ocspdconv/01_base_auto
#
# Note: batch=32 caused CUDA OOM on RTX 4090 for the s model, so the default is
# BATCH=16. You can override it from the command line, e.g. BATCH=8 ./train...

PROJECT_DIR="${PROJECT_DIR:-/home/login/cle/sunfuxiang-1/demo/ours}"
PYTHON_BIN="${PYTHON_BIN:-/home/login/miniconda3/envs/yolo/bin/python}"
MODEL_CFG="${MODEL_CFG:-${PROJECT_DIR}/ultralytics/cfg/models/v8/yolov8s-EnergyGatedOCSPDConv.yaml}"
DATA_CFG="${DATA_CFG:-${PROJECT_DIR}/ultralytics/cfg/datasets/coco8.yaml}"
BASE_CFG="${BASE_CFG:-${PROJECT_DIR}/ultralytics/cfg/default.yaml}"
PROJECT="${PROJECT:-${PROJECT_DIR}/runs/tune_energy_gated_ocspdconv_s}"
NAME="${NAME:-01_base_auto_s_b16}"
BATCH="${BATCH:-16}"

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${TMPDIR:-$HOME/.tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$PROJECT/$NAME"
chmod 700 "$TMPDIR" 2>/dev/null || true

cd "$PROJECT_DIR"

echo "Project: $PROJECT_DIR"
echo "Python:  $PYTHON_BIN"
echo "Model:   $MODEL_CFG"
echo "Data:    $DATA_CFG"
echo "Run dir: $PROJECT/$NAME"
echo "Batch:   $BATCH"

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
from ultralytics import YOLO

project = Path(r"$PROJECT")
name = r"$NAME"
run_dir = project / name
run_dir.mkdir(parents=True, exist_ok=True)

train_kwargs = dict(
    cfg=r"$BASE_CFG",
    data=r"$DATA_CFG",
    epochs=300,
    patience=50,
    batch=int(r"$BATCH"),
    imgsz=640,
    device="0",
    workers=8,
    project=str(project),
    name=name,
    exist_ok=True,
    optimizer="auto",
    lr0=0.01,
    lrf=0.01,
    weight_decay=0.0005,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    warmup_epochs=3.0,
    cos_lr=False,
    mosaic=1.0,
    scale=0.5,
    translate=0.1,
    fliplr=0.5,
    close_mosaic=10,
    seed=82,
    plots=True,
)

model = YOLO(r"$MODEL_CFG")
model.train(**train_kwargs)

best_pt = run_dir / "weights" / "best.pt"
if not best_pt.exists():
    raise FileNotFoundError(f"best.pt not found: {best_pt}")

metrics = YOLO(str(best_pt)).val(
    cfg=r"$BASE_CFG",
    data=r"$DATA_CFG",
    split="test",
    project=str(run_dir),
    name="test",
    exist_ok=True,
    plots=True,
    save_json=False,
    verbose=False,
)

d = metrics.results_dict
result = {
    "name": name,
    "scale": "s",
    "best_pt": str(best_pt),
    "precision": float(d.get("metrics/precision(B)", 0.0)),
    "recall": float(d.get("metrics/recall(B)", 0.0)),
    "map50": float(d.get("metrics/mAP50(B)", 0.0)),
    "map75": float(d.get("metrics/mAP75(B)", 0.0)),
    "map": float(d.get("metrics/mAP50-95(B)", 0.0)),
    "fitness": float(d.get("fitness", 0.0)),
    "params_source": "runs/tune_energy_gated_ocspdconv/01_base_auto",
    "batch": int(r"$BATCH"),
}
result["score"] = 0.50 * result["map"] + 0.25 * result["map50"] + 0.25 * result["map75"]

(run_dir / "test_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
(run_dir / "summary.txt").write_text(
    "EnergyGatedOCSPDConv scale=s with 01_base_auto params\n"
    f"score={result['score']:.10f}\n"
    f"map={result['map']:.10f}\n"
    f"map50={result['map50']:.10f}\n"
    f"map75={result['map75']:.10f}\n"
    f"precision={result['precision']:.10f}\n"
    f"recall={result['recall']:.10f}\n"
    f"fitness={result['fitness']:.10f}\n"
    f"best_pt={result['best_pt']}\n",
    encoding="utf-8",
)
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
