#!/usr/bin/env bash
# Classification 专用运行器：固定使用 conda sunfuxiang 环境的解释器
# 背景：shell 的 ~/bin/python 软链指向 dinov2 环境(Python 3.10,缺依赖)，
#       直接敲 python 会踩坑，本脚本绕开 PATH 问题。
#
# 用法：
#   ./run.sh train  [train_cle_fair.py 的参数...]
#   ./run.sh eval   [evaluate.py 的参数...]
#   ./run.sh python [其他任意 python 脚本/命令]
#
# 示例：
#   ./run.sh train --model resnet34 --num_classes 3 --img_size 224 --seed 42 \
#       --data-path-train ./datasets/cle/train --data-path-val ./datasets/cle/val \
#       --weights /home/login/data/cle/sunfuxiang/classifyML/pth/resnet34-333f7ec4.pth \
#       --save-dir ./weights --epochs 300
#   ./run.sh eval --model resnet34 --weights ./weights/resnet34/model_best.pth \
#       --data-path ./datasets/cle/test
set -euo pipefail
PY="/home/login/miniconda3/envs/sunfuxiang/bin/python"
cd "$(dirname "$0")"

case "${1:-}" in
  train)  shift; exec "$PY" train_cle_fair.py "$@" ;;
  eval)   shift; exec "$PY" evaluate.py "$@" ;;
  python) shift; exec "$PY" "$@" ;;
  *)
    echo "用法: ./run.sh train|eval|python [参数...]"
    exit 1
    ;;
esac
