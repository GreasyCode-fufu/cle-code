#!/bin/bash
# CA 结束后自动接力 CB(同权重根/结果表, 保持 C 轮同批口径)
cd /home/login/data/cle/auto/ZCode/Classification
while pgrep -f "runner.py experiments/jobs_CA.json" > /dev/null; do sleep 30; done
RUN_WEIGHTS=weights_cle_ac_v1_c RUN_RESULTS=experiments/results_cle_ac_v1_c.csv \
  /home/login/miniconda3/envs/sunfuxiang/bin/python experiments/runner.py experiments/jobs_CB.json 4 \
  >> experiments/CB.nohup.log 2>&1
