# convnext_dpea2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_dpea2_seed42/convnext_dpea2/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9839 |
| Accuracy | 0.9834 |
| Macro-AUROC (OvR) | 0.9995 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9834 |
| Balanced Acc | 0.9840 |
| Macro-Sensitivity | 0.9840 |
| Macro-Specificity | 0.9916 |
| Macro-Precision | 0.9840 |
| AUC-PR (Macro) | 0.9991 |
| Cohen's Kappa | 0.9751 |
| MCC | 0.9752 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 28.12M |
| Pure-Inference FPS | 854.6 (1.17 ms/张) |
| End-to-End FPS | 444.1 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 1.0000 | 0.9982 | 0.9959 | 0.9980 | 1.0000 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9668 | 0.9942 | 0.9887 | 0.9776 | 0.9827 | 0.9994 | 0.9988 | 271 |
| PNEUMONIA | 0.9852 | 0.9825 | 0.9673 | 0.9761 | 0.9922 | 0.9992 | 0.9984 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=244 FP=1 TN=540 FN=0
- **NORMAL**: TP=262 FP=3 TN=511 FN=9
- **PNEUMONIA**: TP=266 FP=9 TN=506 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 244 | 0 | 0 |
| NORMAL | 0 | 262 | 9 |
| PNEUMONIA | 1 | 3 | 266 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    1.0000    0.9980       244
      NORMAL     0.9887    0.9668    0.9776       271
   PNEUMONIA     0.9673    0.9852    0.9761       270

    accuracy                         0.9834       785
   macro avg     0.9840    0.9840    0.9839       785
weighted avg     0.9836    0.9834    0.9834       785

```