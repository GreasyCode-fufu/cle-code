#!/bin/bash
# R3 TCGA 单域批: [v4e + convnext_tiny] x seeds 42/43/44 (repo 内 datasets/TCGA 软链)
set -u
cd /home/login/data/cle/auto/ZCode/Classification
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
FINAL=convnext_v4e
classes=$(ls datasets/TCGA/train | sort | tr '\n' ' ')
echo "[tcga] classes= $classes"
python3 - <<'PYEOF'
import json
jobs = [{"model": m, "pth": "convnext_tiny_1k_224_ema.pth", "seed": s}
        for m in ["convnext_v4e", "convnext_tiny"] for s in (42, 43, 44)]
json.dump(jobs, open("experiments/jobs_gen_tcga_r3.json", "w"), indent=1)
PYEOF
RUN_DATA=datasets/TCGA RUN_WEIGHTS=weights_xd_tcga_r3 \
RUN_RESULTS=experiments/results_xd_tcga_r3.csv RUN_NCLS=6 RUN_CLASSES="$classes" \
  $PY experiments/runner.py experiments/jobs_gen_tcga_r3.json 3 >> experiments/gen_tcga_r3.nohup.log 2>&1
echo "[tcga] ALL DONE $(date)"
