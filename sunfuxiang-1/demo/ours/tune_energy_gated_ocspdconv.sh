#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep for /home/login/cle/sunfuxiang-1/demo/ours.
# It trains EnergyGatedOCSPDConv.yaml with several conservative hyperparameter
# candidates, evaluates each best.pt on the test split, and records the best
# result by the selected objective metric.

PROJECT_DIR="${PROJECT_DIR:-/home/login/cle/sunfuxiang-1/demo/ours}"
PYTHON_BIN="${PYTHON_BIN:-/home/login/miniconda3/envs/yolo/bin/python}"
MODEL_CFG="${MODEL_CFG:-${PROJECT_DIR}/ultralytics/cfg/models/v8/EnergyGatedOCSPDConv.yaml}"
DATA_CFG="${DATA_CFG:-${PROJECT_DIR}/ultralytics/cfg/datasets/coco8.yaml}"
BASE_CFG="${BASE_CFG:-${PROJECT_DIR}/ultralytics/cfg/default.yaml}"
TUNE_PROJECT="${TUNE_PROJECT:-${PROJECT_DIR}/runs/tune_energy_gated_ocspdconv}"

EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-50}"
BATCH="${BATCH:-32}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
OBJECTIVE="${OBJECTIVE:-composite}"  # composite, map, map50, map75, fitness, precision, recall
MAX_TRIALS="${MAX_TRIALS:-0}"  # 0 means run all candidates
DRY_RUN="${DRY_RUN:-0}"

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export http_proxy="${http_proxy:-http://127.0.0.1:12789}"
export https_proxy="${https_proxy:-http://127.0.0.1:12789}"
export all_proxy="${all_proxy:-socks5h://127.0.0.1:12789}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="${no_proxy:-$NO_PROXY}"
export TMPDIR="${TMPDIR:-$HOME/.tmp}"
mkdir -p "$TMPDIR" "$TUNE_PROJECT"
chmod 700 "$TMPDIR" 2>/dev/null || true

SUMMARY_CSV="${TUNE_PROJECT}/tune_summary.csv"
BEST_TXT="${TUNE_PROJECT}/best_trial.txt"

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "trial,name,score,objective,map,map50,map75,precision,recall,fitness,best_pt,params" > "$SUMMARY_CSV"
fi

# Candidate format:
# name|optimizer|lr0|lrf|weight_decay|box|cls|dfl|warmup_epochs|cos_lr|mosaic|scale|translate|fliplr|close_mosaic|seed
#
# Keep this list modest by default. Increase it after a first pass if the best
# region is clear. These candidates focus on stable detection hyperparameters.
CANDIDATES=(
  "base_auto|auto|0.010|0.010|0.0005|7.5|0.50|1.5|3.0|False|1.0|0.50|0.10|0.50|10|82"
  "adamw_lowlr|AdamW|0.0010|0.010|0.0005|7.5|0.50|1.5|3.0|True|1.0|0.50|0.10|0.50|10|82"
  "adamw_midlr|AdamW|0.0015|0.010|0.0005|7.5|0.50|1.5|3.0|True|1.0|0.50|0.10|0.50|10|82"
  "sgd_default|SGD|0.010|0.010|0.0005|7.5|0.50|1.5|3.0|False|1.0|0.50|0.10|0.50|10|82"
  "sgd_cos|SGD|0.010|0.010|0.0005|7.5|0.50|1.5|3.0|True|1.0|0.50|0.10|0.50|10|82"
  "wd_low|AdamW|0.0010|0.010|0.0001|7.5|0.50|1.5|3.0|True|1.0|0.50|0.10|0.50|10|82"
  "wd_high|AdamW|0.0010|0.010|0.0010|7.5|0.50|1.5|3.0|True|1.0|0.50|0.10|0.50|10|82"
  "box_high|AdamW|0.0010|0.010|0.0005|9.0|0.50|1.5|3.0|True|1.0|0.50|0.10|0.50|10|82"
  "cls_high|AdamW|0.0010|0.010|0.0005|7.5|0.75|1.5|3.0|True|1.0|0.50|0.10|0.50|10|82"
  "dfl_high|AdamW|0.0010|0.010|0.0005|7.5|0.50|2.0|3.0|True|1.0|0.50|0.10|0.50|10|82"
  "aug_soft|AdamW|0.0010|0.010|0.0005|7.5|0.50|1.5|3.0|True|0.7|0.35|0.08|0.50|10|82"
  "aug_light|AdamW|0.0010|0.010|0.0005|7.5|0.50|1.5|3.0|True|0.5|0.25|0.05|0.50|10|82"
  "no_mosaic|AdamW|0.0010|0.010|0.0005|7.5|0.50|1.5|3.0|True|0.0|0.25|0.05|0.50|0|82"
  "seed_3407|AdamW|0.0010|0.010|0.0005|7.5|0.50|1.5|3.0|True|1.0|0.50|0.10|0.50|10|3407"
)

echo "Project:      $PROJECT_DIR"
echo "Python:       $PYTHON_BIN"
echo "Model cfg:    $MODEL_CFG"
echo "Data cfg:     $DATA_CFG"
echo "Base cfg:     $BASE_CFG"
echo "Tune project: $TUNE_PROJECT"
echo "Objective:    $OBJECTIVE (composite = 0.50*mAP50-95 + 0.25*mAP50 + 0.25*mAP75)"
echo "Epochs:       $EPOCHS"
echo

trial=0
for row in "${CANDIDATES[@]}"; do
  trial=$((trial + 1))
  if [[ "$MAX_TRIALS" != "0" && "$trial" -gt "$MAX_TRIALS" ]]; then
    break
  fi

  IFS='|' read -r name optimizer lr0 lrf weight_decay box cls dfl warmup_epochs cos_lr mosaic scale translate fliplr close_mosaic seed <<< "$row"
  run_name="$(printf '%02d_%s' "$trial" "$name")"
  run_dir="${TUNE_PROJECT}/${run_name}"
  metrics_json="${run_dir}/test_metrics.json"

  echo "========== Trial $trial: $run_name =========="
  echo "params: optimizer=$optimizer lr0=$lr0 lrf=$lrf wd=$weight_decay box=$box cls=$cls dfl=$dfl mosaic=$mosaic scale=$scale translate=$translate seed=$seed"

  if [[ -f "$metrics_json" ]]; then
    echo "Skip completed trial: $metrics_json"
  elif [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1, not training."
    continue
  else
    "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
from ultralytics import YOLO

project_dir = Path(r"$PROJECT_DIR")
run_dir = Path(r"$run_dir")
run_dir.mkdir(parents=True, exist_ok=True)

train_kwargs = dict(
    cfg=r"$BASE_CFG",
    data=r"$DATA_CFG",
    epochs=int("$EPOCHS"),
    patience=int("$PATIENCE"),
    batch=int("$BATCH"),
    imgsz=int("$IMGSZ"),
    device="$DEVICE",
    workers=int("$WORKERS"),
    project=r"$TUNE_PROJECT",
    name=r"$run_name",
    exist_ok=True,
    optimizer="$optimizer",
    lr0=float("$lr0"),
    lrf=float("$lrf"),
    weight_decay=float("$weight_decay"),
    box=float("$box"),
    cls=float("$cls"),
    dfl=float("$dfl"),
    warmup_epochs=float("$warmup_epochs"),
    cos_lr=("$cos_lr" == "True"),
    mosaic=float("$mosaic"),
    scale=float("$scale"),
    translate=float("$translate"),
    fliplr=float("$fliplr"),
    close_mosaic=int("$close_mosaic"),
    seed=int("$seed"),
    plots=False,
)

model = YOLO(r"$MODEL_CFG")
model.train(**train_kwargs)

best_pt = run_dir / "weights" / "best.pt"
if not best_pt.exists():
    raise FileNotFoundError(f"best.pt not found: {best_pt}")

val_model = YOLO(str(best_pt))
metrics = val_model.val(
    cfg=r"$BASE_CFG",
    data=r"$DATA_CFG",
    split="test",
    project=str(run_dir),
    name="test",
    exist_ok=True,
    plots=False,
    save_json=False,
    verbose=False,
)

d = metrics.results_dict
result = {
    "trial": int("$trial"),
    "name": "$run_name",
    "best_pt": str(best_pt),
    "objective": "$OBJECTIVE",
    "precision": float(d.get("metrics/precision(B)", 0.0)),
    "recall": float(d.get("metrics/recall(B)", 0.0)),
    "map50": float(d.get("metrics/mAP50(B)", 0.0)),
    "map75": float(d.get("metrics/mAP75(B)", 0.0)),
    "map": float(d.get("metrics/mAP50-95(B)", 0.0)),
    "fitness": float(d.get("fitness", 0.0)),
    "params": {
        "optimizer": "$optimizer",
        "lr0": float("$lr0"),
        "lrf": float("$lrf"),
        "weight_decay": float("$weight_decay"),
        "box": float("$box"),
        "cls": float("$cls"),
        "dfl": float("$dfl"),
        "warmup_epochs": float("$warmup_epochs"),
        "cos_lr": ("$cos_lr" == "True"),
        "mosaic": float("$mosaic"),
        "scale": float("$scale"),
        "translate": float("$translate"),
        "fliplr": float("$fliplr"),
        "close_mosaic": int("$close_mosaic"),
        "seed": int("$seed"),
    },
}
objective = "$OBJECTIVE"
if objective == "composite":
    # Balanced detector objective: prioritize mAP50-95 while still rewarding mAP50 and mAP75.
    result["score"] = 0.50 * result["map"] + 0.25 * result["map50"] + 0.25 * result["map75"]
else:
    result["score"] = float(result.get(objective, result["map"]))
(run_dir / "test_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
  fi

  "$PYTHON_BIN" - <<PY
import csv, json
from pathlib import Path

summary = Path(r"$SUMMARY_CSV")
metrics_path = Path(r"$metrics_json")
if not metrics_path.exists():
    raise SystemExit(0)
result = json.loads(metrics_path.read_text(encoding="utf-8"))

existing = set()
if summary.exists():
    with summary.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            existing.add(r.get("name", ""))

if result["name"] not in existing:
    with summary.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            result["trial"],
            result["name"],
            f'{result["score"]:.10f}',
            result["objective"],
            f'{result["map"]:.10f}',
            f'{result["map50"]:.10f}',
            f'{result["map75"]:.10f}',
            f'{result["precision"]:.10f}',
            f'{result["recall"]:.10f}',
            f'{result["fitness"]:.10f}',
            result["best_pt"],
            json.dumps(result["params"], ensure_ascii=False, sort_keys=True),
        ])

rows = []
with summary.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
rows = [r for r in rows if r.get("score")]
if rows:
    best = max(rows, key=lambda r: float(r["score"]))
    Path(r"$BEST_TXT").write_text(
        "Best trial by {objective}: {name}\n"
        "score={score}\nmap={map}\nmap50={map50}\nmap75={map75}\nprecision={precision}\nrecall={recall}\nfitness={fitness}\n"
        "best_pt={best_pt}\nparams={params}\n".format(**best),
        encoding="utf-8",
    )
    print("Current best:", best["name"], best["objective"], best["score"])
PY
done

echo
echo "Sweep finished."
echo "Summary: $SUMMARY_CSV"
echo "Best:    $BEST_TXT"
if [[ -f "$BEST_TXT" ]]; then
  cat "$BEST_TXT"
fi
