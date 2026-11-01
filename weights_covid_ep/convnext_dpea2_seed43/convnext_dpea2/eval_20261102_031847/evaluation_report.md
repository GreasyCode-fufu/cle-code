# convnext_dpea2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_dpea2_seed43/convnext_dpea2/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9850 |
| Accuracy | 0.9847 |
| Macro-AUROC (OvR) | 0.9967 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9847 |
| Balanced Acc | 0.9851 |
| Macro-Sensitivity | 0.9851 |
| Macro-Specificity | 0.9923 |
| Macro-Precision | 0.9849 |
| AUC-PR (Macro) | 0.9942 |
| Cohen's Kappa | 0.9770 |
| MCC | 0.9770 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 28.12M |
| Pure-Inference FPS | 310.3 (3.22 ms/张) |
| End-to-End FPS | 230.7 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9963 | 0.9918 | 0.9939 | 0.9981 | 0.9960 | 0.9967 | 244 |
| NORMAL | 0.9889 | 0.9883 | 0.9780 | 0.9834 | 0.9941 | 0.9983 | 0.9969 | 270 |
| PNEUMONIA | 0.9704 | 0.9922 | 0.9850 | 0.9776 | 0.9846 | 0.9958 | 0.9891 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=2 TN=538 FN=1
- **NORMAL**: TP=267 FP=6 TN=508 FN=3
- **PNEUMONIA**: TP=262 FP=4 TN=510 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 267 | 3 |
| PNEUMONIA | 2 | 6 | 262 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9918    0.9959    0.9939       244
      NORMAL     0.9780    0.9889    0.9834       270
   PNEUMONIA     0.9850    0.9704    0.9776       270

    accuracy                         0.9847       784
   macro avg     0.9849    0.9851    0.9850       784
weighted avg     0.9847    0.9847    0.9847       784

```