# 共识消融实验未来执行命令

以下命令是经审计后的执行方案，**本阶段未执行**。它们假定先完成并审查两个新增脚本：

- `scripts/build_consensus_ablation_manifests.py`：只生成/验证 R、V、C、RV、RC、VC manifests；
- `scripts/summarize_consensus_ablation.py`：从统一 predictions 计算并汇总全部指标。

同时假定 `train_cle_fair.py` 已向后兼容地加入 `--train-manifest`，并把完整超参数、输入哈希和实际样本计数写入 `run_meta.json`。在该参数落地前，禁止执行训练段。

## 1. 环境与只读预检查

```bash
set -euo pipefail

PYTHON=/home/login/miniconda3/envs/sunfuxiang/bin/python
REPO=/home/login/data/cle/sunfuxiang/classifyML
DATA_ROOT=/home/login/data/cle/sunfuxiang/datasets
CUFIT=/home/login/data/cle/sunfuxiang/CUFIT0613
MANIFEST_DIR="$REPO/manifests/consensus_ablation"
OUT_ROOT="$REPO/weights_consensus_ablation"
PRETRAINED="$REPO/pth/resnet34-333f7ec4.pth"

cd "$REPO"
test -f "$REPO/train_cle_fair.py"
test -f "$PRETRAINED"
test "$(sha256sum "$PRETRAINED" | awk '{print $1}')" = \
  "333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef"
test "$(find "$DATA_ROOT/3class/train" -type f | wc -l)" -eq 4977
test "$(find "$DATA_ROOT/3class/val" -type f | wc -l)" -eq 538
test "$(find "$DATA_ROOT/3class/test" -type f | wc -l)" -eq 671
diff -qr "$DATA_ROOT/3class/val" "$DATA_ROOT/3class_highquality/val"
diff -qr "$DATA_ROOT/3class/test" "$DATA_ROOT/3class_highquality/test"
```

## 2. 生成并验证六个 manifest（不复制图像）

```bash
"$PYTHON" "$REPO/scripts/build_consensus_ablation_manifests.py" \
  --full-train "$DATA_ROOT/3class/train" \
  --resnet-csv "$REPO/outputs_3class_resnet_newnormal/train_sample_grading.csv" \
  --vit-csv "$REPO/outputs_3class_vit_newnormal/train_sample_grading.csv" \
  --cufit-csv "$CUFIT/cufit_lpm_oof_3class_newnormal.csv" \
  --authoritative-rvc "$CUFIT/consensus3_oof_good_sample_ids.csv" \
  --hq-train "$DATA_ROOT/3class_highquality/train" \
  --out-dir "$MANIFEST_DIR" \
  --expected R=3500 V=3474 C=3749 RV=3103 RC=3141 VC=3115 RVC=2880

"$PYTHON" "$REPO/scripts/build_consensus_ablation_manifests.py" \
  --verify-only \
  --full-train "$DATA_ROOT/3class/train" \
  --manifest-dir "$MANIFEST_DIR" \
  --fixed-val "$DATA_ROOT/3class/val" \
  --fixed-test "$DATA_ROOT/3class/test"
```

## 3. 18 个新训练 run

```bash
set -euo pipefail

PYTHON=/home/login/miniconda3/envs/sunfuxiang/bin/python
REPO=/home/login/data/cle/sunfuxiang/classifyML
DATA_ROOT=/home/login/data/cle/sunfuxiang/datasets
MANIFEST_DIR="$REPO/manifests/consensus_ablation"
OUT_ROOT="$REPO/weights_consensus_ablation"
PRETRAINED="$REPO/pth/resnet34-333f7ec4.pth"

cd "$REPO"

for subset in R V C RV RC VC; do
  for seed in 42 43 44; do
    run_root="$OUT_ROOT/$subset/seed$seed"
    best="$run_root/resnet34/resnet34_${subset}_seed${seed}_best.pth"

    if test -f "$best"; then
      echo "[SKIP-DONE] $best"
      continue
    fi

    "$PYTHON" "$REPO/train_cle_fair.py" \
      --model resnet34 \
      --num_classes 3 \
      --img_size 224 \
      --batch-size 32 \
      --epochs 300 \
      --lr 0.0001 \
      --label-smoothing 0 \
      --seed "$seed" \
      --device cuda:0 \
      --early-stop 30 \
      --weights "$PRETRAINED" \
      --data-path-train "$DATA_ROOT/3class/train" \
      --train-manifest "$MANIFEST_DIR/$subset.csv" \
      --data-path-val "$DATA_ROOT/3class/val" \
      --save-dir "$run_root" \
      --save-name "resnet34_${subset}_seed${seed}_best.pth" \
      --csv-name "train_log_resnet34_${subset}_seed${seed}.csv" \
      --norm-cache-dir "$REPO/norm_cache"
  done
done
```

该循环只包含 R/V/C/RV/RC/VC × 42/43/44，正好 18 runs；不包含 Full 或 RVC，因此不会重训或覆盖旧结果。

## 4. 评估 18 个新模型

```bash
set -euo pipefail

PYTHON=/home/login/miniconda3/envs/sunfuxiang/bin/python
REPO=/home/login/data/cle/sunfuxiang/classifyML
DATA_ROOT=/home/login/data/cle/sunfuxiang/datasets
OUT_ROOT="$REPO/weights_consensus_ablation"

cd "$REPO"

for subset in R V C RV RC VC; do
  for seed in 42 43 44; do
    model_dir="$OUT_ROOT/$subset/seed$seed/resnet34"
    weights="$model_dir/resnet34_${subset}_seed${seed}_best.pth"
    test -f "$weights"

    "$PYTHON" "$REPO/evaluate.py" \
      --model resnet34 \
      --weights "$weights" \
      --data-path "$DATA_ROOT/3class/test" \
      --class-names Benign_lesions Malignant_lesions body_of_stomach_normal \
      --img-size 224 \
      --batch-size 64 \
      --device cuda:0 \
      --out-dir "$model_dir/eval_fixed_test"
  done
done
```

## 5. 统一汇总八个集合

```bash
"$PYTHON" "$REPO/scripts/summarize_consensus_ablation.py" \
  --legacy-map "$REPO/consensus_ablation_audit/LEGACY_PREDICTION_MAP.json" \
  --new-root "$REPO/weights_consensus_ablation" \
  --out-dir "$REPO/consensus_ablation_results"
```

汇总脚本必须为 Full/RVC 显式锁定审计认可的 prediction 文件，并从 `true_label`、`pred_label` 和三类概率统一重算 Accuracy、Macro Precision、Macro Recall、Macro F1、Weighted F1、Macro-AUROC (OvR)。不得把旧 `summary_*.csv` 与新计算结果直接拼表。

## 6. Random-2880（仅未来第二阶段）

以下是设计好的命令接口，本阶段不要运行：

```bash
"$PYTHON" "$REPO/scripts/build_random2880_manifests.py" \
  --full-train "$DATA_ROOT/3class/train" \
  --target-normal 1303 \
  --target-benign 1114 \
  --target-malignant 463 \
  --sampling-seeds 202601 202602 202603 202604 202605 \
  --out-dir "$MANIFEST_DIR/random2880"
```

随后每个 sampling seed 都与训练 seeds 42/43/44 全交叉，共 15 runs；训练参数与上述 18 runs 完全相同，仅 manifest 和输出目录不同。
