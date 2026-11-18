#!/usr/bin/env bash
# =============================================================================
# 批量训练 5 个数据集 (jptomato / insect / strawberry / coffee / raspberry)
# 训练程序: /home/login/cle/sunfuxiang-1/ours/main.py
# 数据集配置: /home/login/cle/sunfuxiang-1/ours/ultralytics/cfg/datasets/coco8.yaml
#
# 说明:
#   - main.py 不接收命令行参数, 只通过 default.yaml 读取配置;
#     因此本脚本通过"重写 coco8.yaml + 设置独立的 name"来逐个切换数据集。
#   - 启动前备份 default.yaml 与 coco8.yaml, 退出(含 Ctrl-C)时自动恢复原文件。
#   - 单个数据集失败不会中断后续数据集; 结束后统一打印 "数据集 -> 权重文件"。
#
# 用法:
#   cd /home/login/cle/sunfuxiang-1/ours
#   ./run_all_datasets.sh
#   # 若训练用特定 conda 环境, 先 activate, 或: export PYTHON=/path/to/python
# =============================================================================
set -uo pipefail

# -------------------------- 路径与可配置项 -----------------------------------
OURS=/home/login/cle/sunfuxiang-1/ours
MAIN=$OURS/main.py
DEFAULT_YAML=$OURS/ultralytics/cfg/default.yaml
COCO8=$OURS/ultralytics/cfg/datasets/coco8.yaml
DATASET_ROOT=/home/login/cle/sunfuxiang-1/dataset
PROJECT_DIR=$OURS/runs/detect          # 权重保存的 project 目录 (绝对路径, 避免歧义)
LOG_DIR=$OURS/train_logs               # 每个数据集的训练日志目录
SUMMARY=$LOG_DIR/weight_map.txt        # 训练汇总文件
PY=${PYTHON:-python}                   # 训练用解释器
EPOCHS=${EPOCHS:-200}                   # 与 RT-DETR 对齐的固定训练轮数(两边都跑完全程)

# -------------------------- 5 个数据集的配置 ---------------------------------
# 顺序: 名称 | nc | names | (train/val/test 由 名称 拼成 $DATASET_ROOT/名称/{train,val,test})
DS_NAME=(jptomato    insect                                              strawberry      coffee                  raspberry)
DS_NC=(   6           13                                                 3               5                       5)
DS_NAMES=(
  "['i', 'a', 'e', 'is', 'q', 't']"
  "['lacewing', 'ladybug_dy', 'stinkbug_hsy', 'stinkbug_mc', 'stinkbug_mx', 'stinkbug_ccc', 'stinkbug_lv', 'stinkbug_zh', 'bollworm', 'ladybug_hb', 'ladybug_lbq', 'syrphid', 'bee']"
  "['i', 'a', 'e']"
  "['is', 'a', 'e', 'i', 'yt']"
  "['is', 'a', 'e', 'i', 'yt']"
)

# -------------------------- 准备: 工作目录 / 日志 / 备份 ---------------------
mkdir -p "$LOG_DIR"
cd "$OURS" || { echo "无法进入 $OURS, 终止"; exit 1; }

BAK_SUFFIX="bak.$$"   # 用 PID 区分, 避免多次运行互相覆盖
cp "$DEFAULT_YAML" "$DEFAULT_YAML.$BAK_SUFFIX" \
  && cp "$COCO8" "$COCO8.$BAK_SUFFIX" \
  || { echo "备份配置文件失败, 终止 (避免污染原始配置)"; exit 1; }

restore_configs() {
  # 无论正常结束、报错还是 Ctrl-C, 都把两个 yaml 还原成运行前的状态
  mv -f "$DEFAULT_YAML.$BAK_SUFFIX" "$DEFAULT_YAML" 2>/dev/null || true
  mv -f "$COCO8.$BAK_SUFFIX" "$COCO8" 2>/dev/null || true
}
trap restore_configs EXIT INT TERM

# 记录结果: 数据集 -> 状态 / 权重路径
declare -A STATUS WEIGHTS

echo "=========================================================="
echo " 批量训练开始  解释器: $PY   (PID=$$)"
echo " 数据集配置: $COCO8"
echo " 权重目录  : $PROJECT_DIR/<数据集>/weights/best.pt"
echo "=========================================================="

# -------------------------- 训练主循环 --------------------------------------
for i in "${!DS_NAME[@]}"; do
  name="${DS_NAME[$i]}"
  nc="${DS_NC[$i]}"
  names="${DS_NAMES[$i]}"
  root="$DATASET_ROOT/$name"
  log="$LOG_DIR/${name}.log"

  echo
  echo "==================== [$((i+1))/${#DS_NAME[@]}] $name  (nc=$nc) 开始训练 ===================="

  # 0) 校验数据集目录
  if [ ! -d "$root/train/images" ] || [ ! -d "$root/val/images" ] || [ ! -d "$root/test/images" ]; then
    echo "[$name] 数据集目录不完整 ($root), 跳过"
    STATUS[$name]="SKIP(目录缺失)"
    WEIGHTS[$name]="-"
    continue
  fi

  # 1) 重写 coco8.yaml 为当前数据集配置
  cat > "$COCO8" <<EOF
# === 由 run_all_datasets.sh 自动生成, 数据集: $name ===
train: $root/train
val: $root/val
test: $root/test

nc: $nc
names: $names
EOF

  # 2) 设置 default.yaml: 修正 data 路径 + 固定 project + 独立 name + 固定 epochs(关早停,与 RT-DETR 对齐)
  #    (data 原值 ours/cfg/... 该目录不存在, 这里改指向真实的 coco8.yaml)
  sed -i \
    -e "s|^data: .*|data: $COCO8|" \
    -e "s|^project: .*|project: $PROJECT_DIR|" \
    -e "s|^name: .*|name: $name|" \
    -e "s|^epochs: .*|epochs: $EPOCHS|" \
    -e "s|^patience: .*|patience: 1000|" \
    "$DEFAULT_YAML"

  echo "[$name] 配置已写入, 开始训练 (日志: $log)"

  # 3) 运行训练 (输出同时落盘 + 终端实时显示; pipefail 保证 if 捕获的是 python 退出码)
  if "$PY" "$MAIN" 2>&1 | tee "$log"; then
    # 4) 定位最新一次的 best.pt (同名重跑会自增后缀, 取最新)
    best=$(ls -dt "$PROJECT_DIR"/${name}*/weights/best.pt 2>/dev/null | head -1)
    if [ -n "$best" ]; then
      STATUS[$name]="OK"
      WEIGHTS[$name]="$best"
      echo "[$name] 训练完成 -> $best"
    else
      STATUS[$name]="OK(未找到权重)"
      WEIGHTS[$name]="(未找到 $PROJECT_DIR/${name}*/weights/best.pt)"
    fi
  else
    STATUS[$name]="FAIL"
    WEIGHTS[$name]="-"
    echo "[$name] 训练失败 (见 $log), 继续下一个数据集"
  fi
done

# -------------------------- 汇总: 数据集 -> 权重文件 ------------------------
echo
echo "=========================================================="
echo " 训练完成汇总 (数据集 -> 权重文件)"
echo "=========================================================="
{
  echo "# 训练汇总  生成于 $(date '+%F %T')  解释器=$PY"
  printf "%-12s %-16s %s\n" "数据集" "状态" "权重文件"
  printf "%-12s %-16s %s\n" "------" "----" "--------"
  for i in "${!DS_NAME[@]}"; do
    name="${DS_NAME[$i]}"
    printf "%-12s %-16s %s\n" "$name" "${STATUS[$name]}" "${WEIGHTS[$name]}"
  done
} | tee "$SUMMARY"

echo
echo "汇总已保存到: $SUMMARY"
echo "原始 default.yaml / coco8.yaml 已自动恢复 (如被 SIGKILL 强杀则不会恢复, 可手动从 *.$BAK_SUFFIX 还原)"
echo "=========================================================="
