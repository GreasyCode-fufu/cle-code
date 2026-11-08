# convnext_v2cema 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v2cema_seed42/convnext_v2cema/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9599 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9957 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9601 |
| Balanced Acc | 0.9597 |
| Macro-Sensitivity | 0.9597 |
| Macro-Specificity | 0.9800 |
| Macro-Precision | 0.9602 |
| AUC-PR (Macro) | 0.9925 |
| Cohen's Kappa | 0.9400 |
| MCC | 0.9400 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1541.7 (0.65 ms/张) |
| End-to-End FPS | 260.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9728 | 0.9526 | 0.9594 | 0.9808 | 0.9940 | 0.9916 | 208 |
| Malignant_lesions | 0.9545 | 0.9775 | 0.9492 | 0.9518 | 0.9799 | 0.9956 | 0.9903 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9896 | 0.9787 | 0.9684 | 0.9794 | 0.9975 | 0.9956 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=10 TN=358 FN=7
- **Malignant_lesions**: TP=168 FP=9 TN=391 FN=8
- **body_of_stomach_normal**: TP=184 FP=4 TN=380 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 5 | 168 | 3 |
| body_of_stomach_normal | 5 | 3 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9526    0.9663    0.9594       208
     Malignant_lesions     0.9492    0.9545    0.9518       176
body_of_stomach_normal     0.9787    0.9583    0.9684       192

              accuracy                         0.9601       576
             macro avg     0.9602    0.9597    0.9599       576
          weighted avg     0.9603    0.9601    0.9601       576

```