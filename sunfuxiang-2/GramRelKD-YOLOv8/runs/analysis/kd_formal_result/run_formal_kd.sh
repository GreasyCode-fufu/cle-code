#!/usr/bin/env bash
set -uo pipefail

project_root="/data/cle/sunfuxiang-2/ultralytics-main"
report_dir="${project_root}/runs/analysis/kd_formal_result"
python_bin="/home/login/miniconda3/envs/yolo/bin/python"
log_path="${report_dir}/formal_train.log"

mkdir -p "${report_dir}"
printf '%s\n' "$$" > "${report_dir}/formal_train.pid"
printf '%s\n' "RUNNING" > "${report_dir}/formal_train.status"
date --iso-8601=seconds > "${report_dir}/formal_train.started_at"

cd "${project_root}" || exit 97
{
  printf 'COMMAND: %s\n' "${python_bin} scripts/distill_train.py --mode train --kd-weight 0.0431658384"
  printf 'EXPECTED_SAVE_DIR: %s\n' "${project_root}/runs/detect/hongkong_SR1_P4Only_RelKD"
  printf 'STARTED_AT: '
  cat "${report_dir}/formal_train.started_at"
} > "${report_dir}/formal_train.launch_metadata.txt"

set +e
"${python_bin}" scripts/distill_train.py \
  --mode train \
  --kd-weight 0.0431658384 \
  2>&1 | tee "${log_path}"
python_status=${PIPESTATUS[0]}
set -e

printf '%s\n' "${python_status}" > "${report_dir}/formal_train.exitcode"
date --iso-8601=seconds > "${report_dir}/formal_train.finished_at"
if [ "${python_status}" -eq 0 ]; then
  printf '%s\n' "SUCCESS" > "${report_dir}/formal_train.status"
else
  printf '%s\n' "FAILED" > "${report_dir}/formal_train.status"
fi
exit "${python_status}"
