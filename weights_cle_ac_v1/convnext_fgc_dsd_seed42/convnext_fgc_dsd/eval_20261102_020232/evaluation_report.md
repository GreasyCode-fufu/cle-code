# convnext_fgc_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_dsd_seed42/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9530 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9946 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9531 |
| Balanced Acc | 0.9530 |
| Macro-Sensitivity | 0.9530 |
| Macro-Specificity | 0.9765 |
| Macro-Precision | 0.9530 |
| AUC-PR (Macro) | 0.9901 |
| Cohen's Kappa | 0.9295 |
| MCC | 0.9295 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 900.0 (1.11 ms/张) |
| End-to-End FPS | 229.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9674 | 0.9423 | 0.9423 | 0.9674 | 0.9918 | 0.9873 | 208 |
| Malignant_lesions | 0.9375 | 0.9725 | 0.9375 | 0.9375 | 0.9725 | 0.9928 | 0.9848 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9896 | 0.9792 | 0.9792 | 0.9896 | 0.9991 | 0.9983 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=12 TN=356 FN=12
- **Malignant_lesions**: TP=165 FP=11 TN=389 FN=11
- **body_of_stomach_normal**: TP=188 FP=4 TN=380 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 10 | 2 |
| Malignant_lesions | 9 | 165 | 2 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9423    0.9423    0.9423       208
     Malignant_lesions     0.9375    0.9375    0.9375       176
body_of_stomach_normal     0.9792    0.9792    0.9792       192

              accuracy                         0.9531       576
             macro avg     0.9530    0.9530    0.9530       576
          weighted avg     0.9531    0.9531    0.9531       576

```