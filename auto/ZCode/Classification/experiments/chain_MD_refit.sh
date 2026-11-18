#!/bin/bash
# R6: MD 批(基线头 + 冻结 /2–/16, 5 seeds) -> 阶段 2 解耦凸头重拟合(三种证据配置) + evaluate.py 评测
cd /home/login/data/cle/auto/ZCode/Classification
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
export RUN_RESULTS=experiments/results_cle_ac_v1_r6.csv RUN_WEIGHTS=weights_cle_ac_v1_r6
$PY experiments/runner.py experiments/jobs_MD.json 4 >> experiments/MD.nohup.log 2>&1
for dst in mbv3_fz16_dce16 mbv3_fz16_dce4s mbv3_fz16_dce4m; do
  $PY experiments/r6_refit.py --src-model mbv3_r6_fz16 --dst-model $dst --seeds 42 43 44 45 46 --batch MDrefit --eval >> experiments/MD.nohup.log 2>&1
done
echo "[chain] MD+refit ALL DONE $(date)" >> experiments/MD.nohup.log
