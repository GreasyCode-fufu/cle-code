# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/vit_base_patch16_224_seed43/vit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8067 |
| Accuracy | 0.8305 |
| Macro-AUROC (OvR) | 0.9578 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8327 |
| Balanced Acc | 0.8275 |
| Macro-Sensitivity | 0.8275 |
| Macro-Specificity | 0.9095 |
| Macro-Precision | 0.8060 |
| AUC-PR (Macro) | 0.9252 |
| Cohen's Kappa | 0.7159 |
| MCC | 0.7205 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 847.3 (1.18 ms/张) |
| End-to-End FPS | 164.9 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8636 | 0.8654 | 0.8906 | 0.8769 | 0.8333 | 0.9384 | 0.9590 | 66 |
| malignant | 0.7188 | 0.9651 | 0.8846 | 0.7931 | 0.9022 | 0.9626 | 0.8905 | 32 |
| normal | 0.9000 | 0.8980 | 0.6429 | 0.7500 | 0.9778 | 0.9724 | 0.9261 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=7 TN=45 FN=9
- **malignant**: TP=23 FP=3 TN=83 FN=9
- **normal**: TP=18 FP=10 TN=88 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 3 | 6 |
| malignant | 5 | 23 | 4 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8906    0.8636    0.8769        66
   malignant     0.8846    0.7188    0.7931        32
      normal     0.6429    0.9000    0.7500        20

    accuracy                         0.8305       118
   macro avg     0.8060    0.8275    0.8067       118
weighted avg     0.8470    0.8305    0.8327       118

```