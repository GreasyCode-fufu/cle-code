# mbv3_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce4s_seed46/mbv3_dce4s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9500 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9913 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9498 |
| Balanced Acc | 0.9498 |
| Macro-Sensitivity | 0.9498 |
| Macro-Specificity | 0.9746 |
| Macro-Precision | 0.9504 |
| AUC-PR (Macro) | 0.9833 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9244 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4257.7 (0.23 ms/张) |
| End-to-End FPS | 12.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9592 | 0.9289 | 0.9356 | 0.9671 | 0.9861 | 0.9755 | 208 |
| Malignant_lesions | 0.9489 | 0.9725 | 0.9382 | 0.9435 | 0.9774 | 0.9893 | 0.9771 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9922 | 0.9840 | 0.9710 | 0.9794 | 0.9986 | 0.9975 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=15 TN=353 FN=12
- **Malignant_lesions**: TP=167 FP=11 TN=389 FN=9
- **body_of_stomach_normal**: TP=184 FP=3 TN=381 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 10 | 2 |
| Malignant_lesions | 8 | 167 | 1 |
| body_of_stomach_normal | 7 | 1 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9289    0.9423    0.9356       208
     Malignant_lesions     0.9382    0.9489    0.9435       176
body_of_stomach_normal     0.9840    0.9583    0.9710       192

              accuracy                         0.9497       576
             macro avg     0.9504    0.9498    0.9500       576
          weighted avg     0.9501    0.9497    0.9498       576

```