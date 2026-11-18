# convnext_fgc_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_fgc_dsd_seed43/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9837 |
| Accuracy | 0.9834 |
| Macro-AUROC (OvR) | 0.9957 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9834 |
| Balanced Acc | 0.9838 |
| Macro-Sensitivity | 0.9838 |
| Macro-Specificity | 0.9916 |
| Macro-Precision | 0.9837 |
| AUC-PR (Macro) | 0.9926 |
| Cohen's Kappa | 0.9751 |
| MCC | 0.9752 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 360.4 (2.77 ms/张) |
| End-to-End FPS | 257.9 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9963 | 0.9918 | 0.9939 | 0.9981 | 0.9965 | 0.9962 | 244 |
| NORMAL | 0.9889 | 0.9864 | 0.9745 | 0.9816 | 0.9941 | 0.9974 | 0.9951 | 270 |
| PNEUMONIA | 0.9667 | 0.9922 | 0.9849 | 0.9757 | 0.9827 | 0.9932 | 0.9865 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=2 TN=538 FN=1
- **NORMAL**: TP=267 FP=7 TN=507 FN=3
- **PNEUMONIA**: TP=261 FP=4 TN=510 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 267 | 3 |
| PNEUMONIA | 2 | 7 | 261 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9918    0.9959    0.9939       244
      NORMAL     0.9745    0.9889    0.9816       270
   PNEUMONIA     0.9849    0.9667    0.9757       270

    accuracy                         0.9834       784
   macro avg     0.9837    0.9838    0.9837       784
weighted avg     0.9835    0.9834    0.9834       784

```