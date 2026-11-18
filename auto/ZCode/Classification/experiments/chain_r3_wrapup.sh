#!/bin/bash
# 等 TCGA(泛化链)全部结束后: 独占GPU收尾(FPS复测+TCGA全指标) + git 提交
cd /home/login/data/cle/auto/ZCode/Classification
while ! grep -q "ALL DONE" experiments/gen_tcga_r3.nohup.log 2>/dev/null; do sleep 120; done
sleep 30
/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/r3_wrapup.py >> experiments/r3_wrapup.nohup.log 2>&1
git add -A
git commit -q -m "实验: R3收尾 — TCGA泛化结果 + 独占GPU复杂度/FPS复测(自动链)"
echo "[wrapup] DONE $(date)"
