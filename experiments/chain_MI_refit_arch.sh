#!/bin/bash
# R6 实验 2 阶段 2(单主干): 4 种重拟合 × 5 seeds + 2 个 LP 臂; 用法: chain_MI_refit_arch.sh <arch> <pth>
cd /home/login/data/cle/auto/ZCode/Classification
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
a=$1; pth=$2; log=experiments/MI_refit_${a}.log
for cfg in "${a}_fz3 ${a}_fz3_dce_last" "${a}_fz3 ${a}_fz3_dce4m" "${a} ${a}_dce_last" "${a} ${a}_dce4m"; do
  set -- $cfg
  $PY experiments/r6_refit.py --src-model $1 --dst-model $2 --seeds 42 43 44 45 46 --batch MIrefit --eval >> $log 2>&1
done
for dst in ${a}_lp_dce4s ${a}_lp_dce4m; do
  $PY experiments/r6_refit.py --src-model imnet --src-ckpt $pth --dst-model $dst --seeds 42 --batch MIrefit --eval >> $log 2>&1
done
echo "[MI refit $a] ALL DONE $(date)" >> $log
