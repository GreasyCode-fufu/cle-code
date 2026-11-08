# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6/mbv3_fz16_dce4m_seed42/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9875 |
| Accuracy | 0.9873 |
| Macro-AUROC (OvR) | 0.9991 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9873 |
| Balanced Acc | 0.9876 |
| Macro-Sensitivity | 0.9876 |
| Macro-Specificity | 0.9936 |
| Macro-Precision | 0.9876 |
| AUC-PR (Macro) | 0.9983 |
| Cohen's Kappa | 0.9809 |
| MCC | 0.9809 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3726.1 (0.27 ms/张) |
| End-to-End FPS | 16.0 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9982 | 0.9959 | 0.9959 | 0.9982 | 0.9999 | 0.9998 | 244 |
| NORMAL | 0.9779 | 0.9942 | 0.9888 | 0.9833 | 0.9884 | 0.9988 | 0.9976 | 271 |
| PNEUMONIA | 0.9889 | 0.9883 | 0.9780 | 0.9834 | 0.9941 | 0.9985 | 0.9975 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=540 FN=1
- **NORMAL**: TP=265 FP=3 TN=511 FN=6
- **PNEUMONIA**: TP=267 FP=6 TN=509 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 1 | 265 | 5 |
| PNEUMONIA | 0 | 3 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9888    0.9779    0.9833       271
   PNEUMONIA     0.9780    0.9889    0.9834       270

    accuracy                         0.9873       785
   macro avg     0.9876    0.9876    0.9875       785
weighted avg     0.9873    0.9873    0.9873       785

```