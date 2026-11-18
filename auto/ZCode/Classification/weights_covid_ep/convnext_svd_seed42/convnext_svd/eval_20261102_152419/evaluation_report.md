# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_svd_seed42/convnext_svd/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9900 |
| Accuracy | 0.9898 |
| Macro-AUROC (OvR) | 0.9995 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9898 |
| Balanced Acc | 0.9900 |
| Macro-Sensitivity | 0.9900 |
| Macro-Specificity | 0.9949 |
| Macro-Precision | 0.9900 |
| AUC-PR (Macro) | 0.9992 |
| Cohen's Kappa | 0.9847 |
| MCC | 0.9847 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 950.6 (1.05 ms/张) |
| End-to-End FPS | 467.7 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9982 | 0.9959 | 0.9959 | 0.9982 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9852 | 0.9961 | 0.9926 | 0.9889 | 0.9922 | 0.9993 | 0.9987 | 271 |
| PNEUMONIA | 0.9889 | 0.9903 | 0.9816 | 0.9852 | 0.9942 | 0.9993 | 0.9988 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=540 FN=1
- **NORMAL**: TP=267 FP=2 TN=512 FN=4
- **PNEUMONIA**: TP=267 FP=5 TN=510 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 267 | 4 |
| PNEUMONIA | 1 | 2 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9926    0.9852    0.9889       271
   PNEUMONIA     0.9816    0.9889    0.9852       270

    accuracy                         0.9898       785
   macro avg     0.9900    0.9900    0.9900       785
weighted avg     0.9898    0.9898    0.9898       785

```