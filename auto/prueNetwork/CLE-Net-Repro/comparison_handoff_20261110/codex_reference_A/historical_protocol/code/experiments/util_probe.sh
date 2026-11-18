#!/bin/bash
# 利用率/吞吐探针: 跑 N 个 epoch, 期间每秒采样 GPU, 结束报告 s/epoch
# 用法: util_probe.sh <tag> <epochs> <num_workers> <并发数>
set -u
TAG=$1; EPOCHS=${2:-4}; NW=${3:-3}; CONC=${4:-1}; MODEL=${5:-convnext_egb_scg_hsc}
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
cd /home/login/data/cle/auto/Network/CLE-Net || exit 1
OUT=/tmp/util_${TAG}
rm -rf ${OUT}; mkdir -p ${OUT}

# 后台 GPU 采样(1s)
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used \
  --format=csv,noheader,nounits -l 1 > ${OUT}/gpu.csv 2>/dev/null &
GPU_SAMPLER=$!
# 后台 CPU 采样
( while true; do
    awk '{print $1, $2, $3, $4, $5}' /proc/stat >> ${OUT}/cpu.csv
    sleep 1
  done ) &
CPU_SAMPLER=$!

START=$(date +%s)
TRAIN_PIDS=""
for i in $(seq 1 ${CONC}); do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ${PY} train_cle_fair.py \
    --model ${MODEL} --num_classes 3 --img_size 224 --seed 4${i} \
    --data-path-train ./datasets/cle_ac_v1/train \
    --data-path-val ./datasets/cle_ac_v1/val \
    --weights ./pretrained/convnext_tiny_1k_224_ema.pth \
    --save-dir /tmp/probe_util_${TAG} --epochs ${EPOCHS} --num-workers ${NW} \
    > ${OUT}/train${i}.log 2>&1 &
  TRAIN_PIDS="${TRAIN_PIDS} $!"
done
for p in ${TRAIN_PIDS}; do wait $p; done
END=$(date +%s)
kill ${GPU_SAMPLER} ${CPU_SAMPLER} 2>/dev/null

python3 - "$OUT" "$TAG" "$EPOCHS" "$CONC" "$((END-START))" <<'PYEOF'
import sys, statistics as st
out, tag, epochs, conc, wall = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
gpu = []
for line in open(out + "/gpu.csv"):
    p = [x.strip() for x in line.split(",")]
    if len(p) >= 3:
        try:
            gpu.append((float(p[0]), float(p[1]), float(p[2])))
        except ValueError:
            pass
# CPU: 用相邻快照差算总利用率
rows = []
for line in open(out + "/cpu.csv"):
    p = line.split()
    if len(p) == 5 and p[0] == "cpu":
        rows.append([int(x) for x in p[1:]])
cpu_util = []
for a, b in zip(rows, rows[1:]):
    da = [y - x for x, y in zip(a, b)]
    tot = sum(da)
    if tot > 0:
        idle = da[3] + (da[4] if len(da) > 4 else 0)
        cpu_util.append(100.0 * (tot - idle) / tot)
print(f"[{tag}] wall={wall}s  epochs={epochs}x{conc}  -> {wall/ (epochs*conc):.1f}s/epoch(等效)")
if gpu:
    # 去掉前 15s warmup
    g = gpu[15:] if len(gpu) > 20 else gpu
    print(f"[{tag}] GPU util: mean={st.mean(x[0] for x in g):.1f}%  "
          f"p50={st.median(x[0] for x in g):.0f}%  max={max(x[0] for x in g):.0f}%  "
          f"mem_util={st.mean(x[1] for x in g):.1f}%  VRAM={st.mean(x[2] for x in g):.0f}MiB")
    print(f"[{tag}] GPU util 分布: " + " ".join(
        f"{lo}-{lo+20}:{sum(1 for x in g if lo <= x[0] < lo+20)}" for lo in range(0, 100, 20)))
if cpu_util:
    print(f"[{tag}] CPU util: mean={st.mean(cpu_util):.1f}%  max={max(cpu_util):.0f}% "
          f"(24 核, 满载=2400%)")
PYEOF
grep -h "train epoch" ${OUT}/train*.log | tail -1
