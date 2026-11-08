# mbv3_r6_s2_lsside 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_lsside_seed43/mbv3_r6_s2_lsside/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9644 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9980 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9720 |
| Balanced Acc | 0.9669 |
| Macro-Sensitivity | 0.9669 |
| Macro-Specificity | 0.9863 |
| Macro-Precision | 0.9621 |
| AUC-PR (Macro) | 0.9938 |
| Cohen's Kappa | 0.9554 |
| MCC | 0.9555 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 235.94M |
| FLOPs | 471.88M |
| Params | 3.27M |
| Pure-Inference FPS | 2249.9 (0.44 ms/张) |
| End-to-End FPS | 122.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9641 | 0.9828 | 0.9792 | 0.9716 | 0.9702 | 0.9979 | 0.9976 | 195 |
| Malignant_lesions | 0.9367 | 0.9799 | 0.9136 | 0.9250 | 0.9855 | 0.9961 | 0.9838 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=4 TN=228 FN=7
- **Malignant_lesions**: TP=74 FP=7 TN=341 FN=5
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 7 | 0 |
| Malignant_lesions | 4 | 74 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9792    0.9641    0.9716       195
     Malignant_lesions     0.9136    0.9367    0.9250        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9719       427
             macro avg     0.9621    0.9669    0.9644       427
          weighted avg     0.9722    0.9719    0.9720       427

```