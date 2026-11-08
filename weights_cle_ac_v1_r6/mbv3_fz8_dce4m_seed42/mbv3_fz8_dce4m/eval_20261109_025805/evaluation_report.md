# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz8_dce4m_seed42/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9618 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9971 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9618 |
| Balanced Acc | 0.9608 |
| Macro-Sensitivity | 0.9608 |
| Macro-Specificity | 0.9806 |
| Macro-Precision | 0.9630 |
| AUC-PR (Macro) | 0.9944 |
| Cohen's Kappa | 0.9425 |
| MCC | 0.9427 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4326.7 (0.23 ms/张) |
| End-to-End FPS | 17.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9620 | 0.9349 | 0.9504 | 0.9806 | 0.9956 | 0.9928 | 208 |
| Malignant_lesions | 0.9318 | 0.9850 | 0.9647 | 0.9480 | 0.9704 | 0.9959 | 0.9910 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9948 | 0.9895 | 0.9869 | 0.9922 | 0.9997 | 0.9994 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=14 TN=354 FN=7
- **Malignant_lesions**: TP=164 FP=6 TN=394 FN=12
- **body_of_stomach_normal**: TP=189 FP=2 TN=382 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 12 | 164 | 0 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9349    0.9663    0.9504       208
     Malignant_lesions     0.9647    0.9318    0.9480       176
body_of_stomach_normal     0.9895    0.9844    0.9869       192

              accuracy                         0.9618       576
             macro avg     0.9630    0.9608    0.9618       576
          weighted avg     0.9622    0.9618    0.9618       576

```