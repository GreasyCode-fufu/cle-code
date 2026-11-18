# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/jumbo_small_seed42/jumbo_small/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9802 |
| Accuracy | 0.9796 |
| Macro-AUROC (OvR) | 0.9986 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9796 |
| Balanced Acc | 0.9800 |
| Macro-Sensitivity | 0.9800 |
| Macro-Specificity | 0.9896 |
| Macro-Precision | 0.9805 |
| AUC-PR (Macro) | 0.9976 |
| Cohen's Kappa | 0.9694 |
| MCC | 0.9695 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 904.2 (1.11 ms/张) |
| End-to-End FPS | 454.2 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 0.9998 | 0.9996 | 244 |
| NORMAL | 0.9852 | 0.9786 | 0.9604 | 0.9727 | 0.9921 | 0.9977 | 0.9963 | 271 |
| PNEUMONIA | 0.9630 | 0.9903 | 0.9811 | 0.9720 | 0.9808 | 0.9983 | 0.9968 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=267 FP=11 TN=503 FN=4
- **PNEUMONIA**: TP=260 FP=5 TN=510 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 267 | 4 |
| PNEUMONIA | 0 | 10 | 260 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9604    0.9852    0.9727       271
   PNEUMONIA     0.9811    0.9630    0.9720       270

    accuracy                         0.9796       785
   macro avg     0.9805    0.9800    0.9802       785
weighted avg     0.9799    0.9796    0.9796       785

```