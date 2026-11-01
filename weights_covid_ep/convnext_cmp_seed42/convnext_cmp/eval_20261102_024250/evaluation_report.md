# convnext_cmp 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_cmp_seed42/convnext_cmp/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9875 |
| Accuracy | 0.9873 |
| Macro-AUROC (OvR) | 0.9992 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9873 |
| Balanced Acc | 0.9875 |
| Macro-Sensitivity | 0.9875 |
| Macro-Specificity | 0.9936 |
| Macro-Precision | 0.9875 |
| AUC-PR (Macro) | 0.9985 |
| Cohen's Kappa | 0.9809 |
| MCC | 0.9809 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 468.6 (2.13 ms/张) |
| End-to-End FPS | 310.6 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9982 | 0.9959 | 0.9959 | 0.9982 | 0.9998 | 0.9996 | 244 |
| NORMAL | 0.9852 | 0.9922 | 0.9852 | 0.9852 | 0.9922 | 0.9991 | 0.9983 | 271 |
| PNEUMONIA | 0.9815 | 0.9903 | 0.9815 | 0.9815 | 0.9903 | 0.9987 | 0.9975 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=540 FN=1
- **NORMAL**: TP=267 FP=4 TN=510 FN=4
- **PNEUMONIA**: TP=265 FP=5 TN=510 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 267 | 4 |
| PNEUMONIA | 1 | 4 | 265 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9852    0.9852    0.9852       271
   PNEUMONIA     0.9815    0.9815    0.9815       270

    accuracy                         0.9873       785
   macro avg     0.9875    0.9875    0.9875       785
weighted avg     0.9873    0.9873    0.9873       785

```