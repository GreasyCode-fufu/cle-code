# convnext_v2c 评估报告

- 权重: `weights_tmp/soups/v2c_soup3.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: imagenet_fallback  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9607 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9978 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9614 |
| Balanced Acc | 0.9594 |
| Macro-Sensitivity | 0.9594 |
| Macro-Specificity | 0.9805 |
| Macro-Precision | 0.9640 |
| AUC-PR (Macro) | 0.9965 |
| Cohen's Kappa | 0.9425 |
| MCC | 0.9433 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1668.7 (0.60 ms/张) |
| End-to-End FPS | 226.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9856 | 0.9674 | 0.9447 | 0.9647 | 0.9916 | 0.9959 | 0.9952 | 208 |
| Malignant_lesions | 0.8977 | 0.9950 | 0.9875 | 0.9405 | 0.9567 | 0.9982 | 0.9957 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9792 | 0.9598 | 0.9770 | 0.9973 | 0.9993 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=205 FP=12 TN=356 FN=3
- **Malignant_lesions**: TP=158 FP=2 TN=398 FN=18
- **body_of_stomach_normal**: TP=191 FP=8 TN=376 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 205 | 2 | 1 |
| Malignant_lesions | 11 | 158 | 7 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9447    0.9856    0.9647       208
     Malignant_lesions     0.9875    0.8977    0.9405       176
body_of_stomach_normal     0.9598    0.9948    0.9770       192

              accuracy                         0.9618       576
             macro avg     0.9640    0.9594    0.9607       576
          weighted avg     0.9628    0.9618    0.9614       576

```