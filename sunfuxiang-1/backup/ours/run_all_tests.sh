#!/usr/bin/env bash
# =============================================================================
# 对 run_all_datasets.sh 产生的 8 个 best.pt 依次在各自 test 集上评估。
# 输出目录: /home/login/cle/sunfuxiang-1/ours/runs/detect/test_eval/<模型_数据集>
# 日志目录: /home/login/cle/sunfuxiang-1/ours/test_logs
# =============================================================================
set -uo pipefail

OURS=/home/login/cle/sunfuxiang-1/ours
RUNS_DIR=$OURS/runs/detect
DATA_YAML_DIR=$OURS/train_logs/dataset_yamls
TEST_PROJECT=$RUNS_DIR/test_eval
LOG_DIR=$OURS/test_logs
SUMMARY=$LOG_DIR/test_summary.txt
PY=${PYTHON:-python}

MODEL_NAME=(ablation-2 test)
DS_NAME=(jptomato raspberry strawberry tomato)

mkdir -p "$LOG_DIR" "$TEST_PROJECT"
cd "$OURS" || { echo "无法进入 $OURS，终止"; exit 1; }

declare -A STATUS RESULTS

echo "=========================================================="
echo "开始测试：${#MODEL_NAME[@]} 个模型 × ${#DS_NAME[@]} 个数据集"
echo "解释器: $PY"
echo "结果目录: $TEST_PROJECT"
echo "=========================================================="

for model in "${MODEL_NAME[@]}"; do
  for dataset in "${DS_NAME[@]}"; do
    key="${model}|${dataset}"
    run_name="${model}_${dataset}"
    weight="$RUNS_DIR/$run_name/weights/best.pt"
    data_yaml="$DATA_YAML_DIR/$run_name.yaml"
    log="$LOG_DIR/$run_name.log"

    echo
    echo "==================== [$model] [$dataset] 开始测试 ===================="

    if [ ! -f "$weight" ]; then
      echo "权重不存在：$weight"
      STATUS[$key]="SKIP(权重缺失)"
      RESULTS[$key]="-"
      continue
    fi
    if [ ! -f "$data_yaml" ]; then
      echo "数据 YAML 不存在：$data_yaml"
      STATUS[$key]="SKIP(数据YAML缺失)"
      RESULTS[$key]="-"
      continue
    fi

    # 使用每项训练时保留的数据 YAML，确保 split='test' 指向正确的数据集。
    if "$PY" - "$weight" "$data_yaml" "$TEST_PROJECT" "$run_name" 2>&1 <<'PY' | tee "$log"
import sys
from ultralytics import YOLO

weight, data, project, name = sys.argv[1:]
metrics = YOLO(weight).val(data=data, split="test", project=project, name=name, exist_ok=True)

# 不同 Ultralytics 版本的指标属性略有差异，因此只输出稳定的结果对象。
print("TEST_METRICS:", metrics)
PY
    then
      STATUS[$key]="OK"
      RESULTS[$key]="$TEST_PROJECT/$run_name"
      echo "[$model][$dataset] 测试完成 -> ${RESULTS[$key]}"
    else
      STATUS[$key]="FAIL"
      RESULTS[$key]="-"
      echo "[$model][$dataset] 测试失败（见 $log），继续下一项"
    fi
  done
done

echo
echo "=========================================================="
echo "测试完成汇总"
echo "=========================================================="
{
  echo "# 生成于 $(date '+%F %T') 解释器=$PY"
  printf "%-12s %-12s %-18s %s\n" "模型" "数据集" "状态" "结果目录"
  for model in "${MODEL_NAME[@]}"; do
    for dataset in "${DS_NAME[@]}"; do
      key="${model}|${dataset}"
      printf "%-12s %-12s %-18s %s\n" "$model" "$dataset" "${STATUS[$key]:-UNKNOWN}" "${RESULTS[$key]:--}"
    done
  done
} | tee "$SUMMARY"

echo
echo "汇总已保存到: $SUMMARY"
