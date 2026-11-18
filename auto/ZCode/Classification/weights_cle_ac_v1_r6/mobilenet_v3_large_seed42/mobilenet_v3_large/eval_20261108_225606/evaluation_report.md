# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mobilenet_v3_large_seed42/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9494 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9941 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9496 |
| Balanced Acc | 0.9489 |
| Macro-Sensitivity | 0.9489 |
| Macro-Specificity | 0.9746 |
| Macro-Precision | 0.9502 |
| AUC-PR (Macro) | 0.9883 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9244 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 4683.7 (0.21 ms/张) |
| End-to-End FPS | 239.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9620 | 0.9336 | 0.9403 | 0.9699 | 0.9910 | 0.9856 | 208 |
| Malignant_lesions | 0.9205 | 0.9800 | 0.9529 | 0.9364 | 0.9655 | 0.9920 | 0.9808 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9818 | 0.9641 | 0.9716 | 0.9895 | 0.9992 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=14 TN=354 FN=11
- **Malignant_lesions**: TP=162 FP=8 TN=392 FN=14
- **body_of_stomach_normal**: TP=188 FP=7 TN=377 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 8 | 3 |
| Malignant_lesions | 10 | 162 | 4 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9336    0.9471    0.9403       208
     Malignant_lesions     0.9529    0.9205    0.9364       176
body_of_stomach_normal     0.9641    0.9792    0.9716       192

              accuracy                         0.9497       576
             macro avg     0.9502    0.9489    0.9494       576
          weighted avg     0.9497    0.9497    0.9496       576

```