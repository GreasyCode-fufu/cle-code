#!/usr/bin/env bash
set -uo pipefail

project=/data/cle/sunfuxiang-2/ultralytics-main
report="$project/runs/analysis/mtrd_formal_result"
python=/home/login/miniconda3/envs/yolo/bin/python
cd "$project" || exit 97

echo RUNNING > "$report/formal_test.status"
date -Is > "$report/formal_test.started_at"
$python scripts/test_mtrd.py 2>&1 | tee "$report/formal_test.log"
official_status=${PIPESTATUS[0]}
echo "$official_status" > "$report/formal_test.exitcode"
date -Is > "$report/formal_test.finished_at"
if [ "$official_status" -ne 0 ]; then
  echo FAILED > "$report/formal_test.status"
  exit "$official_status"
fi

$python scripts/extract_mtrd_metrics.py 2>&1 | tee "$report/formal_test_metrics_extract.log"
extract_status=${PIPESTATUS[0]}
echo "$extract_status" > "$report/formal_test_metrics_extract.exitcode"
if [ "$extract_status" -eq 0 ]; then
  echo SUCCESS > "$report/formal_test.status"
else
  echo FAILED > "$report/formal_test.status"
fi
exit "$extract_status"
