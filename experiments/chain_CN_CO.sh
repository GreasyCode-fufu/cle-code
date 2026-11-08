#!/bin/bash
cd /home/login/data/cle/auto/ZCode/Classification
while kill -0 4070130 2>/dev/null; do sleep 20; done
RUN_WEIGHTS=weights_cle_ac_v1_c RUN_RESULTS=experiments/results_cle_ac_v1_c.csv   /home/login/miniconda3/envs/sunfuxiang/bin/python experiments/runner.py experiments/jobs_CO.json 4 >> experiments/CO.nohup.log 2>&1
