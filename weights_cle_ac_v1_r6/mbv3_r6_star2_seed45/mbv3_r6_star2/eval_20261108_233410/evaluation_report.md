# mbv3_r6_star2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star2_seed45/mbv3_r6_star2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9367 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9890 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9371 |
| Balanced Acc | 0.9345 |
| Macro-Sensitivity | 0.9345 |
| Macro-Specificity | 0.9679 |
| Macro-Precision | 0.9426 |
| AUC-PR (Macro) | 0.9791 |
| Cohen's Kappa | 0.9058 |
| MCC | 0.9074 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 3963.7 (0.25 ms/张) |
| End-to-End FPS | 220.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9321 | 0.8894 | 0.9263 | 0.9800 | 0.9826 | 0.9687 | 208 |
| Malignant_lesions | 0.8580 | 0.9900 | 0.9742 | 0.9124 | 0.9406 | 0.9858 | 0.9719 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9818 | 0.9641 | 0.9716 | 0.9895 | 0.9985 | 0.9968 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=25 TN=343 FN=7
- **Malignant_lesions**: TP=151 FP=4 TN=396 FN=25
- **body_of_stomach_normal**: TP=188 FP=7 TN=377 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 4 | 3 |
| Malignant_lesions | 21 | 151 | 4 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8894    0.9663    0.9263       208
     Malignant_lesions     0.9742    0.8580    0.9124       176
body_of_stomach_normal     0.9641    0.9792    0.9716       192

              accuracy                         0.9375       576
             macro avg     0.9426    0.9345    0.9367       576
          weighted avg     0.9402    0.9375    0.9371       576

```