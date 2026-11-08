# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce4s_seed45/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9620 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9950 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9619 |
| Balanced Acc | 0.9622 |
| Macro-Sensitivity | 0.9622 |
| Macro-Specificity | 0.9808 |
| Macro-Precision | 0.9620 |
| AUC-PR (Macro) | 0.9905 |
| Cohen's Kappa | 0.9426 |
| MCC | 0.9426 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 5786.4 (0.17 ms/张) |
| End-to-End FPS | 235.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9701 | 0.9471 | 0.9471 | 0.9701 | 0.9923 | 0.9893 | 208 |
| Malignant_lesions | 0.9602 | 0.9750 | 0.9441 | 0.9521 | 0.9824 | 0.9934 | 0.9833 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9974 | 0.9947 | 0.9869 | 0.9897 | 0.9994 | 0.9988 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=11 TN=357 FN=11
- **Malignant_lesions**: TP=169 FP=10 TN=390 FN=7
- **body_of_stomach_normal**: TP=188 FP=1 TN=383 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 7 | 169 | 0 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9471    0.9471    0.9471       208
     Malignant_lesions     0.9441    0.9602    0.9521       176
body_of_stomach_normal     0.9947    0.9792    0.9869       192

              accuracy                         0.9618       576
             macro avg     0.9620    0.9622    0.9620       576
          weighted avg     0.9621    0.9618    0.9619       576

```