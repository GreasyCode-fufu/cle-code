#!/usr/bin/env bash
set -o pipefail
cd /data/cle/sunfuxiang-2/ultralytics-main
exec /home/login/miniconda3/envs/yolo/bin/python scripts/run_nightly_kd_sweep.py
