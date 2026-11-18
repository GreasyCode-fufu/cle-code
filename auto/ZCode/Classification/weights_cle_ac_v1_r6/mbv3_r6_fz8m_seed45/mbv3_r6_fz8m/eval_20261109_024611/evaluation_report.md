# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz8m_seed45/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9617 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9941 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9619 |
| Balanced Acc | 0.9609 |
| Macro-Sensitivity | 0.9609 |
| Macro-Specificity | 0.9807 |
| Macro-Precision | 0.9628 |
| AUC-PR (Macro) | 0.9900 |
| Cohen's Kappa | 0.9425 |
| MCC | 0.9427 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3045.0 (0.33 ms/张) |
| End-to-End FPS | 207.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9647 | 0.9395 | 0.9551 | 0.9834 | 0.9901 | 0.9873 | 208 |
| Malignant_lesions | 0.9375 | 0.9825 | 0.9593 | 0.9483 | 0.9728 | 0.9936 | 0.9859 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9948 | 0.9894 | 0.9816 | 0.9871 | 0.9985 | 0.9967 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=13 TN=355 FN=6
- **Malignant_lesions**: TP=165 FP=7 TN=393 FN=11
- **body_of_stomach_normal**: TP=187 FP=2 TN=382 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 11 | 165 | 0 |
| body_of_stomach_normal | 2 | 3 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9395    0.9712    0.9551       208
     Malignant_lesions     0.9593    0.9375    0.9483       176
body_of_stomach_normal     0.9894    0.9740    0.9816       192

              accuracy                         0.9618       576
             macro avg     0.9628    0.9609    0.9617       576
          weighted avg     0.9622    0.9618    0.9619       576

```