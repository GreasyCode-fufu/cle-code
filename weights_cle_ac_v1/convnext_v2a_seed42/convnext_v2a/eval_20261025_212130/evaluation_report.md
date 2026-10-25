# convnext_v2a 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2a_seed42/convnext_v2a/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9598 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9601 |
| Balanced Acc | 0.9600 |
| Macro-Sensitivity | 0.9600 |
| Macro-Specificity | 0.9800 |
| Macro-Precision | 0.9597 |
| AUC-PR (Macro) | 0.9926 |
| Cohen's Kappa | 0.9400 |
| MCC | 0.9400 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.47G |
| FLOPs | 8.95G |
| Params | 27.81M |
| Pure-Inference FPS | 641.8 (1.56 ms/张) |
| End-to-End FPS | 166.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9728 | 0.9517 | 0.9494 | 0.9702 | 0.9950 | 0.9909 | 208 |
| Malignant_lesions | 0.9432 | 0.9725 | 0.9379 | 0.9405 | 0.9749 | 0.9937 | 0.9880 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9948 | 0.9896 | 0.9896 | 0.9948 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=10 TN=358 FN=11
- **Malignant_lesions**: TP=166 FP=11 TN=389 FN=10
- **body_of_stomach_normal**: TP=190 FP=2 TN=382 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 9 | 166 | 1 |
| body_of_stomach_normal | 1 | 1 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9517    0.9471    0.9494       208
     Malignant_lesions     0.9379    0.9432    0.9405       176
body_of_stomach_normal     0.9896    0.9896    0.9896       192

              accuracy                         0.9601       576
             macro avg     0.9597    0.9600    0.9598       576
          weighted avg     0.9601    0.9601    0.9601       576

```