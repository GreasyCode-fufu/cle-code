#!/usr/bin/env bash
# =============================================================================
# 批量训练脚本：3class 数据质量对照实验（多 seed 版：42 / 43 / 44）
#   高质量集(三方共识 2880 张) vs 全量(4977 张)，同模型同超参，
#   都加载 ImageNet 预训练权重微调。3 个 seed 用于报均值±标准差（论文级）。
#
# 训练脚本 train_cle_fair.py：加载预训练权重 → 自动用 ImageNet 归一化；
#   两数据集都加载预训练，故都走 ImageNet 归一化，唯一差异 = 训练集大小/纯度。
#
# 仅训练能干净加载预训练权重的 6 个模型（densenet121 已移除）。
#
# 规模：6 模型 × 2 数据集 × 3 seed = 36 个训练任务。
#
# 产物命名（seed 进目录名 + 文件名，三维度可区分，便于评估脚本通配汇总）：
#   weights_<dataset>_seed<seed>/<model>/<model>_<dataset>_seed<seed>_best.pth
#   例: weights_3class_hq_seed43/resnet34/resnet34_3class_hq_seed43_best.pth
#       weights_3class_full_seed44/vit_base_patch16_224/..._best.pth
#   每个 run 目录另有: run_meta.json / train_log_*.csv / class_indices.json
#
# 断点友好：若某 (seed×dataset×model) 的 best 权重已存在则跳过，不重训。
#   → 重跑本脚本只会补缺失的（如已有 seed42，本次只跑 43/44）。
# =============================================================================
set -u   # 未定义变量报错；故意不用 set -e，单个失败不影响其余

# ----------------------------- 可调配置 -----------------------------
REPO_DIR="/home/login/cle/sunfuxiang/classifyML"
PTH_DIR="${REPO_DIR}/pth"

EPOCHS=300
BATCH_SIZE=32
IMG_SIZE=224
LR=0.0001
DEVICE="cuda:0"
EARLY_STOP=30
NORM_CACHE="${REPO_DIR}/norm_cache"

# ---------- 三个随机种子 ----------
declare -a SEEDS=(42 43 44)

# ---------- 模型注册名 | 预训练权重文件名 ----------
declare -a MODELS=(
  "fast_t0|fasternet_t0-epoch.281-val_acc1.71.9180.pth"
  "resnet34|resnet34-333f7ec4.pth"
  "vit_base_patch16_224|vit_base_patch16_224.pth"
  "swin_tiny_patch4_window7_224|swin_tiny_patch4_window7_224.pth"
  "pvt_tiny|pvt_tiny.pth"
  "repvit_m1|repvit_m1_0_distill_300e.pth"
)

# ---------- 数据集: 名称 | 类别数 | 根目录 ----------
declare -a DATASETS=(
  "3class_hq|3|/home/login/cle/sunfuxiang/datasets/3class_highquality"
  "3class_full|3|/home/login/cle/sunfuxiang/datasets/3class"
)
# -------------------------------------------------------------------


# 训练单个 (seed × dataset × model) 组合
train_one() {
  local seed="$1" ds_name="$2" num_classes="$3" ds_root="$4"
  local model="$5" weight_file="$6"

  local weight_path="${PTH_DIR}/${weight_file}"
  local save_dir="${REPO_DIR}/weights_${ds_name}_seed${seed}"
  local save_name="${model}_${ds_name}_seed${seed}_best.pth"
  local final_weight="${save_dir}/${model}/${save_name}"

  # 跳过已完成：best 权重已存在则不重训
  if [ -f "${final_weight}" ]; then
    echo "[SKIP-DONE] seed=${seed} ${model}@${ds_name} 已存在 -> ${final_weight}"
    return 0
  fi

  # 前置检查
  if [ ! -f "${weight_path}" ]; then
    echo "[SKIP] 预训练权重不存在: ${weight_path}"
    return 1
  fi
  if [ ! -d "${ds_root}/train" ] || [ ! -d "${ds_root}/val" ]; then
    echo "[SKIP] 数据集目录缺失 train/val: ${ds_root}"
    return 1
  fi

  mkdir -p "${save_dir}"

  echo "================================================================"
  echo "[TRAIN] seed=${seed}  dataset=${ds_name} (nc=${num_classes})  model=${model}"
  echo "        pretrained : ${weight_path}"
  echo "        norm       : ImageNet (加载预训练，自动切换)"
  echo "        early-stop : ${EARLY_STOP}  epochs(max ${EPOCHS})"
  echo "        out weights: ${final_weight}"
  echo "================================================================"

  python "${REPO_DIR}/train_cle_fair.py" \
    --model "${model}" \
    --num_classes "${num_classes}" \
    --img_size "${IMG_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --early-stop "${EARLY_STOP}" \
    --weights "${weight_path}" \
    --data-path-train "${ds_root}/train" \
    --data-path-val   "${ds_root}/val" \
    --save-dir "${save_dir}" \
    --save-name "${save_name}" \
    --csv-name "train_log_${model}_${ds_name}_seed${seed}.csv" \
    --norm-cache-dir "${NORM_CACHE}"

  local rc=$?
  if [ ${rc} -ne 0 ]; then
    echo "[FAIL] seed=${seed} ${model}@${ds_name} (exit ${rc})"
    return ${rc}
  fi

  cp -f "class_indices.json" "${save_dir}/${model}/class_indices.json" 2>/dev/null || true
  echo "[OK]   seed=${seed} ${model}@${ds_name} -> ${final_weight}"
  echo ""
  return 0
}

# ------------------------------- 主流程 -------------------------------
main() {
  cd "${REPO_DIR}" || { echo "无法进入 ${REPO_DIR}"; exit 1; }

  echo "Repo         : ${REPO_DIR}"
  echo "Pretrained   : ${PTH_DIR}"
  echo "Seeds        : ${SEEDS[*]}"
  echo "Ep/Bs/Im/Lr/ES : ${EPOCHS} / ${BATCH_SIZE} / ${IMG_SIZE} / ${LR} / earlystop=${EARLY_STOP}"
  echo "Datasets     : $(for d in "${DATASETS[@]}"; do echo "${d%%|*}"; done | tr '\n' ' ')"
  echo "Models       : $(for m in "${MODELS[@]}"; do echo "${m%%|*}"; done | tr '\n' ' ')"
  echo "对照实验     : 高质量集(2880) vs 全量(4977)，6 模型 × 2 数据集 × 3 seed = 36 个任务"
  echo

  local failed=() skipped_done=0 total=0
  # seed 外层：先把一个 seed 全跑完，结果尽早可用
  for seed in "${SEEDS[@]}"; do
    echo "########## SEED ${seed} ##########"
    for ds in "${DATASETS[@]}"; do
      IFS='|' read -r ds_name num_classes ds_root <<< "${ds}"
      for entry in "${MODELS[@]}"; do
        IFS='|' read -r model weight_file <<< "${entry}"
        total=$((total + 1))
        if ! train_one "${seed}" "${ds_name}" "${num_classes}" "${ds_root}" "${model}" "${weight_file}"; then
          failed+=("seed${seed}:${model}@${ds_name}")
        fi
      done
    done
  done

  echo "################################################################"
  echo "# 完成: 共 ${total} 个任务槽位（含已跳过的已完成项）"
  if [ ${#failed[@]} -gt 0 ]; then
    echo "# 失败 (${#failed[@]}): ${failed[*]}"
  else
    echo "# 无失败 ✔"
  fi
  echo "# 权重目录: ${REPO_DIR}/weights_3class_{hq,full}_seed{42,43,44}/"
  echo "################################################################"
}

main "$@"