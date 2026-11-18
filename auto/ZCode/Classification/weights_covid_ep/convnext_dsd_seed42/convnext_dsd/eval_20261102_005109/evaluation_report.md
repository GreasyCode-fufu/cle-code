# convnext_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_dsd_seed42/convnext_dsd/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9851 |
| Accuracy | 0.9847 |
| Macro-AUROC (OvR) | 0.9993 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9847 |
| Balanced Acc | 0.9851 |
| Macro-Sensitivity | 0.9851 |
| Macro-Specificity | 0.9922 |
| Macro-Precision | 0.9853 |
| AUC-PR (Macro) | 0.9986 |
| Cohen's Kappa | 0.9770 |
| MCC | 0.9771 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 27.78M |
| Pure-Inference FPS | 764.1 (1.31 ms/张) |
| End-to-End FPS | 415.7 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 1.0000 | 1.0000 | 0.9979 | 0.9982 | 0.9999 | 0.9998 | 244 |
| NORMAL | 0.9889 | 0.9844 | 0.9710 | 0.9799 | 0.9941 | 0.9991 | 0.9981 | 271 |
| PNEUMONIA | 0.9704 | 0.9922 | 0.9850 | 0.9776 | 0.9846 | 0.9988 | 0.9979 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=0 TN=541 FN=1
- **NORMAL**: TP=268 FP=8 TN=506 FN=3
- **PNEUMONIA**: TP=262 FP=4 TN=511 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 268 | 3 |
| PNEUMONIA | 0 | 8 | 262 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9959    0.9979       244
      NORMAL     0.9710    0.9889    0.9799       271
   PNEUMONIA     0.9850    0.9704    0.9776       270

    accuracy                         0.9847       785
   macro avg     0.9853    0.9851    0.9851       785
weighted avg     0.9848    0.9847    0.9847       785

```