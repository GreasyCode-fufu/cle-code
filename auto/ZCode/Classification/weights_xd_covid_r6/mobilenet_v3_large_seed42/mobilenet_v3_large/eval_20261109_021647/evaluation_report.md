# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6/mobilenet_v3_large_seed42/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9888 |
| Accuracy | 0.9885 |
| Macro-AUROC (OvR) | 0.9976 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9885 |
| Balanced Acc | 0.9888 |
| Macro-Sensitivity | 0.9888 |
| Macro-Specificity | 0.9942 |
| Macro-Precision | 0.9888 |
| AUC-PR (Macro) | 0.9958 |
| Cohen's Kappa | 0.9828 |
| MCC | 0.9828 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 3903.8 (0.26 ms/张) |
| End-to-End FPS | 611.9 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9970 | 0.9971 | 244 |
| NORMAL | 0.9889 | 0.9922 | 0.9852 | 0.9871 | 0.9942 | 0.9989 | 0.9980 | 270 |
| PNEUMONIA | 0.9815 | 0.9922 | 0.9851 | 0.9833 | 0.9903 | 0.9969 | 0.9922 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=267 FP=4 TN=510 FN=3
- **PNEUMONIA**: TP=265 FP=4 TN=510 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 267 | 3 |
| PNEUMONIA | 1 | 4 | 265 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9852    0.9889    0.9871       270
   PNEUMONIA     0.9851    0.9815    0.9833       270

    accuracy                         0.9885       784
   macro avg     0.9888    0.9888    0.9888       784
weighted avg     0.9885    0.9885    0.9885       784

```