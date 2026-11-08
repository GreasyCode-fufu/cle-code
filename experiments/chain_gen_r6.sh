#!/bin/bash
# R6 泛化: 5 域 × {baseline mobilenet_v3_large, 阶段1 mbv3_r6_fz16(原 MLP 头 + 冻结 /2–/16)} × 3 seeds 同批同协议;
# 每域训练后做阶段 2 重拟合(C 仅在该域 val 上选):
#   fz16 -> mbv3_fz16_dce16(①+②) / mbv3_fz16_dce4s(①+②+多尺度) / mbv3_fz16_dce4m(①+②+③ = 最终网络 EPIC-MobileNet)
#   baseline -> mbv3_dce4m(②+③, 无①; 检验冻结在各域是否必要)
# 跨域消融链: baseline → ① fz16 → ①+② fz16_dce16 → ①+②+③ fz16_dce4m。小域先、TCGA 最后。
set -u
cd /home/login/data/cle/auto/ZCode/Classification
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
run_ds () { local name=$1 data=$2 ncls=$3 conc=$4
  local jobs=experiments/jobs_gen6_${name}.json log=experiments/gen6_${name}.nohup.log
  python3 - "$jobs" <<'PYEOF'
import json, sys
json.dump([{"model": m, "pth": "mobilenet_v3_large-8738ca79.pth", "seed": s}
           for m in ("mobilenet_v3_large", "mbv3_r6_fz16") for s in (42, 43, 44)], open(sys.argv[1], "w"), indent=1)
PYEOF
  local classes; classes=$(ls "$data/train" | sort | tr '\n' ' ')
  export RUN_DATA="$data" RUN_WEIGHTS="weights_xd_${name}_r6" RUN_RESULTS="experiments/results_xd_${name}_r6.csv" \
         RUN_NCLS=$ncls RUN_CLASSES="$classes"
  echo "[gen6:$name] stage1 start $(date)" >> $log
  $PY experiments/runner.py "$jobs" "$conc" >> $log 2>&1
  for spec in "mbv3_r6_fz16 mbv3_fz16_dce16" "mbv3_r6_fz16 mbv3_fz16_dce4s" "mbv3_r6_fz16 mbv3_fz16_dce4m" \
              "mobilenet_v3_large mbv3_dce4m"; do
    set -- $spec
    $PY experiments/r6_refit.py --src-model $1 --dst-model $2 --seeds 42 43 44 --src-root "$RUN_WEIGHTS" \
      --dst-root "$RUN_WEIGHTS" --data "$data" --results "$RUN_RESULTS" --batch gen6refit --eval >> $log 2>&1
  done
  echo "[gen6:$name] ALL DONE $(date)" >> $log
}
run_ds busi  datasets/busi_v1  3   4
run_ds heel  datasets/heelBone 3   4
run_ds covid datasets/covid_v1 3   4
run_ds cub   datasets/cub_v1   200 3
run_ds tcga  datasets/TCGA     6   3
echo "[gen6] ALL DONE $(date)" >> experiments/gen6_chain.nohup.log
