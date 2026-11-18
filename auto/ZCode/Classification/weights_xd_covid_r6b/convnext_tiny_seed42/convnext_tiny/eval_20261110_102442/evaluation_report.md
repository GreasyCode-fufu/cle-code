# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/convnext_tiny_seed42/convnext_tiny/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9826 |
| Accuracy | 0.9822 |
| Macro-AUROC (OvR) | 0.9989 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9822 |
| Balanced Acc | 0.9824 |
| Macro-Sensitivity | 0.9824 |
| Macro-Specificity | 0.9909 |
| Macro-Precision | 0.9830 |
| AUC-PR (Macro) | 0.9977 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9733 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 645.4 (1.55 ms/张) |
| End-to-End FPS | 378.5 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9877 | 1.0000 | 1.0000 | 0.9938 | 0.9945 | 0.9997 | 0.9994 | 244 |
| NORMAL | 0.9705 | 0.9922 | 0.9850 | 0.9777 | 0.9846 | 0.9980 | 0.9955 | 271 |
| PNEUMONIA | 0.9889 | 0.9806 | 0.9639 | 0.9762 | 0.9941 | 0.9990 | 0.9982 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=241 FP=0 TN=541 FN=3
- **NORMAL**: TP=263 FP=4 TN=510 FN=8
- **PNEUMONIA**: TP=267 FP=10 TN=505 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 241 | 1 | 2 |
| NORMAL | 0 | 263 | 8 |
| PNEUMONIA | 0 | 3 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9877    0.9938       244
      NORMAL     0.9850    0.9705    0.9777       271
   PNEUMONIA     0.9639    0.9889    0.9762       270

    accuracy                         0.9822       785
   macro avg     0.9830    0.9824    0.9826       785
weighted avg     0.9824    0.9822    0.9822       785

```