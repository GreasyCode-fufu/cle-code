#!/bin/bash
# R6 实验 2 阶段 2: 等 MI runner(PID 453664)结束 -> 各主干 6 种重拟合 + evaluate.py
cd /home/login/data/cle/auto/ZCode/Classification
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
P=/home/login/data/cle/sunfuxiang/classifyML/pth
while kill -0 453664 2>/dev/null; do sleep 20; done
for spec in "convnext_tiny $P/convnext_tiny_1k_224_ema.pth" "starnet_s2 $P/timm_r6/starnet_s2.pth" "fast_t0 $P/fasternet_t0-epoch.281-val_acc1.71.9180.pth"; do
  set -- $spec; a=$1; pth=$2
  for cfg in "${a}_fz3 ${a}_fz3_dce_last" "${a}_fz3 ${a}_fz3_dce4m" "${a} ${a}_dce_last" "${a} ${a}_dce4m"; do
    set -- $cfg
    $PY experiments/r6_refit.py --src-model $1 --dst-model $2 --seeds 42 43 44 45 46 --batch MIrefit --eval >> experiments/MI_refit.log 2>&1
  done
  for dst in ${a}_lp_dce4s ${a}_lp_dce4m; do
    $PY experiments/r6_refit.py --src-model imnet --src-ckpt $pth --dst-model $dst --seeds 42 --batch MIrefit --eval >> experiments/MI_refit.log 2>&1
  done
done
echo "[MI refit] ALL DONE $(date)" >> experiments/MI_refit.log
