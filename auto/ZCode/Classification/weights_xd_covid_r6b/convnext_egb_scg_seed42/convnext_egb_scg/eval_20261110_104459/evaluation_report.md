# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/convnext_egb_scg_seed42/convnext_egb_scg/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9838 |
| Accuracy | 0.9834 |
| Macro-AUROC (OvR) | 0.9971 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9834 |
| Balanced Acc | 0.9838 |
| Macro-Sensitivity | 0.9838 |
| Macro-Specificity | 0.9916 |
| Macro-Precision | 0.9839 |
| AUC-PR (Macro) | 0.9946 |
| Cohen's Kappa | 0.9751 |
| MCC | 0.9752 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 344.6 (2.90 ms/张) |
| End-to-End FPS | 251.4 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9966 | 0.9972 | 244 |
| NORMAL | 0.9889 | 0.9844 | 0.9709 | 0.9798 | 0.9941 | 0.9985 | 0.9974 | 270 |
| PNEUMONIA | 0.9667 | 0.9922 | 0.9849 | 0.9757 | 0.9827 | 0.9963 | 0.9892 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=267 FP=8 TN=506 FN=3
- **PNEUMONIA**: TP=261 FP=4 TN=510 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 267 | 3 |
| PNEUMONIA | 1 | 8 | 261 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9709    0.9889    0.9798       270
   PNEUMONIA     0.9849    0.9667    0.9757       270

    accuracy                         0.9834       784
   macro avg     0.9839    0.9838    0.9838       784
weighted avg     0.9835    0.9834    0.9834       784

```