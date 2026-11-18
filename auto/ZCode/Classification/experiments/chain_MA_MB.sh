#!/bin/bash
# R6 接力: 等 MA runner(显式 PID)结束 -> 启动 MB 批
cd /home/login/data/cle/auto/ZCode/Classification
while kill -0 6240 2>/dev/null; do sleep 20; done
RUN_RESULTS=experiments/results_cle_ac_v1_r6.csv RUN_WEIGHTS=weights_cle_ac_v1_r6 \
  /home/login/miniconda3/envs/sunfuxiang/bin/python experiments/runner.py experiments/jobs_MB.json 4 >> experiments/MB.nohup.log 2>&1
