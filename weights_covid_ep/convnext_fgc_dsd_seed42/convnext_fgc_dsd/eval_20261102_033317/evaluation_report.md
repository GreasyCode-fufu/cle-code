# convnext_fgc_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_fgc_dsd_seed42/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9814 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9993 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9809 |
| Balanced Acc | 0.9813 |
| Macro-Sensitivity | 0.9813 |
| Macro-Specificity | 0.9903 |
| Macro-Precision | 0.9819 |
| AUC-PR (Macro) | 0.9988 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9715 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 266.5 (3.75 ms/张) |
| End-to-End FPS | 205.2 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 0.9998 | 0.9996 | 244 |
| NORMAL | 0.9631 | 0.9942 | 0.9886 | 0.9757 | 0.9808 | 0.9992 | 0.9984 | 271 |
| PNEUMONIA | 0.9889 | 0.9767 | 0.9570 | 0.9727 | 0.9941 | 0.9990 | 0.9983 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=261 FP=3 TN=511 FN=10
- **PNEUMONIA**: TP=267 FP=12 TN=503 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 0 | 2 |
| NORMAL | 0 | 261 | 10 |
| PNEUMONIA | 0 | 3 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9886    0.9631    0.9757       271
   PNEUMONIA     0.9570    0.9889    0.9727       270

    accuracy                         0.9809       785
   macro avg     0.9819    0.9813    0.9814       785
weighted avg     0.9813    0.9809    0.9809       785

```