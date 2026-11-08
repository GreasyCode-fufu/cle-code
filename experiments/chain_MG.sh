#!/bin/bash
# R6 MG: ① 的温和形式(fz16bn: 冻结 /2–/16 卷积核+BN 自适应; fz8m: 只冻结 /2–/8)在 CLE(5 seeds) 与冻结有害域 BUSI/heelBone(3 seeds)
# 上的阶段 1 + 阶段 2(DCE 4 尺度 FOV 掩码) —— 检验能否得到跨域通用的"证据保持"。等 MF runner(PID 208809)结束再启动。
cd /home/login/data/cle/auto/ZCode/Classification
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
while kill -0 208809 2>/dev/null; do sleep 20; done
mk () { python3 -c "import json,sys; json.dump([{'model':m,'pth':'mobilenet_v3_large-8738ca79.pth','seed':s} for m in ('mbv3_r6_fz16bn','mbv3_r6_fz8m') for s in sys.argv[2:]], open(sys.argv[1],'w'), indent=1)" "$@"; }
run () { local name=$1 data=$2 ncls=$3 wroot=$4 res=$5; shift 5; local seeds="$*"
  mk experiments/jobs_MG_${name}.json $seeds
  export RUN_DATA="$data" RUN_WEIGHTS="$wroot" RUN_RESULTS="$res" RUN_NCLS=$ncls RUN_CLASSES="$(ls $data/train | sort | tr '\n' ' ')"
  [ "$name" = cle ] && unset RUN_CLASSES
  $PY experiments/runner.py experiments/jobs_MG_${name}.json 3 >> experiments/MG.nohup.log 2>&1
  for spec in "mbv3_r6_fz16bn mbv3_fz16bn_dce4m" "mbv3_r6_fz8m mbv3_fz8_dce4m"; do set -- $spec
    $PY experiments/r6_refit.py --src-model $1 --dst-model $2 --seeds $seeds --src-root "$wroot" --dst-root "$wroot" \
      --data "$data" --results "$res" --batch MGrefit --eval >> experiments/MG.nohup.log 2>&1
  done
}
run cle  datasets/cle_ac_v1 3 weights_cle_ac_v1_r6 experiments/results_cle_ac_v1_r6.csv 42 43 44 45 46
run busi datasets/busi_v1   3 weights_xd_busi_r6   experiments/results_xd_busi_r6.csv   42 43 44
run heel datasets/heelBone  3 weights_xd_heel_r6   experiments/results_xd_heel_r6.csv   42 43 44
echo "[MG] ALL DONE $(date)" >> experiments/MG.nohup.log
