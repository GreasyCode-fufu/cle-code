#!/bin/bash
# ==============================================================================
# CLE Fair Baseline Training Script
# Trains 4 models (vit_base_patch16_224, swin_tiny_patch4_window7_224,
#                  pvt_tiny, repvit_m1) × 3 seeds (42, 43, 44)
# Total: 12 training runs
# ==============================================================================

# ---------------- Configuration ----------------
# Data paths
DATA_PATH_TRAIN="/home/login/cle/sunfuxiang/datasets/2class/train"
DATA_PATH_VAL="/home/login/cle/sunfuxiang/datasets/2class/val"

# Common training parameters
NUM_CLASSES=2
IMG_SIZE=224
BATCH_SIZE=32
LR=0.0001
EPOCHS=300
EARLY_STOP=50
DEVICE="cuda:0"

# Directory settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTH_DIR="${SCRIPT_DIR}/pth"
SAVE_DIR="${SCRIPT_DIR}/weights_fair"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_cle_fair.py"

# Create output directory
mkdir -p "${SAVE_DIR}"

# Model and pretrained weights mapping
declare -A MODELS
MODELS["vit_base_patch16_224"]="${PTH_DIR}/vit_base_patch16_224.pth"
MODELS["swin_tiny_patch4_window7_224"]="${PTH_DIR}/swin_tiny_patch4_window7_224.pth"
MODELS["pvt_tiny"]="${PTH_DIR}/pvt_tiny.pth"
MODELS["repvit_m1"]="${PTH_DIR}/repvit_m1_0_distill_300e.pth"

# Seeds to run
SEEDS=(42 43 44)

# ==============================================================================

echo "========================================"
echo "CLE Fair Baseline Training"
echo "========================================"
echo "Train Data: ${DATA_PATH_TRAIN}"
echo "Val Data:   ${DATA_PATH_VAL}"
echo "Save Dir:   ${SAVE_DIR}"
echo "Early Stop: ${EARLY_STOP} epochs"
echo "========================================"

# Check if train script exists
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "ERROR: Training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

# Counter for total runs
TOTAL_RUNS=0
CURRENT_RUN=0

# Calculate total runs
for model in "${!MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ((TOTAL_RUNS++))
    done
done

echo "Total training runs: ${TOTAL_RUNS}"
echo ""

# ==================== Training Loop ====================
for model in "${!MODELS[@]}"; do
    weights="${MODELS[$model]}"

    # Check if weights file exists
    if [ ! -f "${weights}" ]; then
        echo "WARNING: Weights file not found: ${weights}"
        echo "Skipping ${model}..."
        continue
    fi

    for seed in "${SEEDS[@]}"; do
        ((CURRENT_RUN++))

        # Output weight filename: model+seed (e.g., fast_t042.pth)
        save_name="${model}${seed}.pth"

        # Skip if already trained (check if output weight exists)
        existing=$(find "${SAVE_DIR}" -name "${save_name}" 2>/dev/null | head -1)
        if [ -n "${existing}" ]; then
            echo "========================================"
            echo "[${CURRENT_RUN}/${TOTAL_RUNS}] SKIP: ${model} seed=${seed} already exists"
            echo "  Found: ${existing}"
            echo "========================================"
            echo ""
            continue
        fi

        echo "========================================"
        echo "[${CURRENT_RUN}/${TOTAL_RUNS}] Training: ${model} with seed=${seed}"
        echo "========================================"
        echo "  Model:      ${model}"
        echo "  Seed:       ${seed}"
        echo "  Weights:    ${weights}"
        echo "  Save name:  ${save_name}"
        echo "----------------------------------------"

        # Run training
        python3 "${TRAIN_SCRIPT}" \
            --model "${model}" \
            --seed "${seed}" \
            --num_classes "${NUM_CLASSES}" \
            --img_size "${IMG_SIZE}" \
            --batch-size "${BATCH_SIZE}" \
            --lr "${LR}" \
            --epochs "${EPOCHS}" \
            --early-stop "${EARLY_STOP}" \
            --data-path-train "${DATA_PATH_TRAIN}" \
            --data-path-val "${DATA_PATH_VAL}" \
            --weights "${weights}" \
            --device "${DEVICE}" \
            --save-dir "${SAVE_DIR}" \
            --save-name "${save_name}" \
            --csv-name "train_${model}_${seed}.csv"

        EXIT_CODE=$?
        if [ ${EXIT_CODE} -eq 0 ]; then
            echo ">>> Training completed: ${save_name}"
        else
            echo ">>> ERROR: Training failed with exit code ${EXIT_CODE}"
        fi
        echo ""
    done
done

echo "========================================"
echo "All training runs completed!"
echo "Results saved in: ${SAVE_DIR}"
echo "========================================"
