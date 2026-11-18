# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz8_dce4m_seed44/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9600 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9967 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9601 |
| Balanced Acc | 0.9597 |
| Macro-Sensitivity | 0.9597 |
| Macro-Specificity | 0.9799 |
| Macro-Precision | 0.9603 |
| AUC-PR (Macro) | 0.9937 |
| Cohen's Kappa | 0.9399 |
| MCC | 0.9400 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 5371.1 (0.19 ms/张) |
| End-to-End FPS | 20.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9701 | 0.9476 | 0.9522 | 0.9754 | 0.9951 | 0.9919 | 208 |
| Malignant_lesions | 0.9432 | 0.9800 | 0.9540 | 0.9486 | 0.9751 | 0.9960 | 0.9910 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9896 | 0.9792 | 0.9792 | 0.9896 | 0.9992 | 0.9983 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=11 TN=357 FN=9
- **Malignant_lesions**: TP=166 FP=8 TN=392 FN=10
- **body_of_stomach_normal**: TP=188 FP=4 TN=380 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 8 | 166 | 2 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9476    0.9567    0.9522       208
     Malignant_lesions     0.9540    0.9432    0.9486       176
body_of_stomach_normal     0.9792    0.9792    0.9792       192

              accuracy                         0.9601       576
             macro avg     0.9603    0.9597    0.9600       576
          weighted avg     0.9601    0.9601    0.9601       576

```