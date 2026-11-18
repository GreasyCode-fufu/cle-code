# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_ef_seed43/convnext_ef/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9838 |
| Accuracy | 0.9834 |
| Macro-AUROC (OvR) | 0.9993 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9835 |
| Balanced Acc | 0.9837 |
| Macro-Sensitivity | 0.9837 |
| Macro-Specificity | 0.9916 |
| Macro-Precision | 0.9841 |
| AUC-PR (Macro) | 0.9987 |
| Cohen's Kappa | 0.9751 |
| MCC | 0.9752 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 523.1 (1.91 ms/张) |
| End-to-End FPS | 332.7 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9982 | 0.9959 | 0.9938 | 0.9963 | 0.9999 | 0.9999 | 244 |
| NORMAL | 0.9705 | 0.9961 | 0.9925 | 0.9813 | 0.9846 | 0.9990 | 0.9984 | 271 |
| PNEUMONIA | 0.9889 | 0.9806 | 0.9639 | 0.9762 | 0.9941 | 0.9989 | 0.9980 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=1 TN=540 FN=2
- **NORMAL**: TP=263 FP=2 TN=512 FN=8
- **PNEUMONIA**: TP=267 FP=10 TN=505 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 0 | 2 |
| NORMAL | 0 | 263 | 8 |
| PNEUMONIA | 1 | 2 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9918    0.9938       244
      NORMAL     0.9925    0.9705    0.9813       271
   PNEUMONIA     0.9639    0.9889    0.9762       270

    accuracy                         0.9834       785
   macro avg     0.9841    0.9837    0.9838       785
weighted avg     0.9837    0.9834    0.9835       785

```