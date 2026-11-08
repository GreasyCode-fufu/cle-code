# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_abl/convnext_v2m_seed43/convnext_v2m/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9838 |
| Accuracy | 0.9834 |
| Macro-AUROC (OvR) | 0.9996 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9835 |
| Balanced Acc | 0.9837 |
| Macro-Sensitivity | 0.9837 |
| Macro-Specificity | 0.9916 |
| Macro-Precision | 0.9840 |
| AUC-PR (Macro) | 0.9992 |
| Cohen's Kappa | 0.9751 |
| MCC | 0.9752 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 862.8 (1.16 ms/张) |
| End-to-End FPS | 443.5 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9982 | 0.9959 | 0.9938 | 0.9963 | 1.0000 | 1.0000 | 244 |
| NORMAL | 0.9705 | 0.9942 | 0.9887 | 0.9795 | 0.9846 | 0.9993 | 0.9987 | 271 |
| PNEUMONIA | 0.9889 | 0.9825 | 0.9674 | 0.9780 | 0.9941 | 0.9994 | 0.9989 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=1 TN=540 FN=2
- **NORMAL**: TP=263 FP=3 TN=511 FN=8
- **PNEUMONIA**: TP=267 FP=9 TN=506 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 263 | 8 |
| PNEUMONIA | 1 | 2 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9918    0.9938       244
      NORMAL     0.9887    0.9705    0.9795       271
   PNEUMONIA     0.9674    0.9889    0.9780       270

    accuracy                         0.9834       785
   macro avg     0.9840    0.9837    0.9838       785
weighted avg     0.9836    0.9834    0.9835       785

```