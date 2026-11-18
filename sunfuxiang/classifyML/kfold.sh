#!/usr/bin/env bash
# smoke_resnet_fixed.sh — ResNet34 单方案短验证,专看 best_epoch 是否落在健康区间
# 3分类 / 1 round / 5 fold / 50 epoch,约 15-20 分钟(ResNet 比 ViT 快)
set -euo pipefail

REPO="/home/login/cle/sunfuxiang/classifyML"
PY="/home/login/miniconda3/envs/sunfuxiang/bin/python"
CSV="/home/login/cle/sunfuxiang/csv/train_processed.csv"
WEIGHTS="/home/login/cle/sunfuxiang/classifyML/pth/resnet34-333f7ec4.pth"
TRAIN="/home/login/cle/sunfuxiang/datasets/3class/train"
OUT="./outputs_smoke_resnet_v2"

cd "$REPO"
mkdir -p "$OUT"
LOG="$OUT/smoke_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo " ResNet34 短验证(检验 best_epoch 是否健康)"
echo " 方案: 3分类 1round×5fold 50epoch"
echo " 权重: resnet34-333f7ec4.pth (ImageNet)"
echo " monitor=macro_f1  min_epochs=10  patience=10"
echo " 日志: $LOG"
echo "========================================"

stdbuf -oL -eL "$PY" KFold.py \
  --model-module models.Resnet --model-function resnet34 \
  --weights "$WEIGHTS" \
  --data-path-train "$TRAIN" \
  --csv-path "$CSV" \
  --num-classes 3 --num-folds 5 --num-rounds 1 \
  --img-size 224 --batch-size 32 --lr 1e-4 \
  --epochs 50 \
  --use-early-stop --early-stop-monitor macro_f1 \
  --early-stop-patience 10 --early-stop-min-epochs 10 \
  --output-dir "$OUT/quality_grading" \
  --output-csv "$OUT/train_sample_grading.csv" \
  --patient-summary-csv "$OUT/train_patient_grading_summary.csv" \
  2>&1 | tee "$LOG"

echo ""
echo "========================================"
echo " 权重加载核对(主干应全部加载,只 fc missing):"
grep "\[权重\]" "$LOG" | head -6
echo ""
echo " best_epoch 分布:"
grep "恢复" "$LOG" | grep -oE "best_epoch=[0-9]+" | sort | uniq -c
echo ""
echo " 各折峰值 macro_f1:"
grep "早停" "$LOG" | grep -oE "best_macro_f1=[0-9.]+"
echo "========================================"
echo " 判断: best_epoch 普遍 20-50 且峰值F1明显>0.57 → 换模型成功,可上全量"
echo "       best_epoch 仍 1-3 → 根因不在模型,需另查(贴回给我)"