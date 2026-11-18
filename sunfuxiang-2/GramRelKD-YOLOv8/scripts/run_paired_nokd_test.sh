#!/usr/bin/env bash
set -uo pipefail

project_root="/data/cle/sunfuxiang-2/ultralytics-main"
report_dir="${project_root}/runs/analysis/kd_paired_nokd_result"
python_bin="/home/login/miniconda3/envs/yolo/bin/python"

cd "${project_root}" || exit 97
printf '%s\n' "RUNNING" > "${report_dir}/paired_nokd_test.status"
date --iso-8601=seconds > "${report_dir}/paired_nokd_test.started_at"
set +e
"${python_bin}" scripts/test_paired_nokd.py 2>&1 | tee "${report_dir}/paired_nokd_test.log"
python_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "${python_status}" > "${report_dir}/paired_nokd_test.exitcode"
date --iso-8601=seconds > "${report_dir}/paired_nokd_test.finished_at"
if [ "${python_status}" -eq 0 ]; then
  printf '%s\n' "SUCCESS" > "${report_dir}/paired_nokd_test.status"
else
  printf '%s\n' "FAILED" > "${report_dir}/paired_nokd_test.status"
fi
exit "${python_status}"
