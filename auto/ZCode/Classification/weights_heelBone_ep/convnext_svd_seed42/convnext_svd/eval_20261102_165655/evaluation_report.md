# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_svd_seed42/convnext_svd/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8685 |
| Accuracy | 0.8634 |
| Macro-AUROC (OvR) | 0.9753 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8626 |
| Balanced Acc | 0.8840 |
| Macro-Sensitivity | 0.8840 |
| Macro-Specificity | 0.9293 |
| Macro-Precision | 0.8579 |
| AUC-PR (Macro) | 0.9644 |
| Cohen's Kappa | 0.7870 |
| MCC | 0.7886 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 1085.9 (0.92 ms/张) |
| End-to-End FPS | 278.2 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8426 | 0.9217 | 0.8426 | 0.8426 | 0.9217 | 0.9661 | 0.9456 | 197 |
| normal | 0.8261 | 0.9148 | 0.8941 | 0.8588 | 0.8580 | 0.9619 | 0.9544 | 276 |
| sever | 0.9833 | 0.9514 | 0.8369 | 0.9042 | 0.9956 | 0.9980 | 0.9931 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=166 FP=31 TN=365 FN=31
- **normal**: TP=228 FP=27 TN=290 FN=48
- **sever**: TP=118 FP=23 TN=450 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 166 | 25 | 6 |
| normal | 31 | 228 | 17 |
| sever | 0 | 2 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8426    0.8426    0.8426       197
      normal     0.8941    0.8261    0.8588       276
       sever     0.8369    0.9833    0.9042       120

    accuracy                         0.8634       593
   macro avg     0.8579    0.8840    0.8685       593
weighted avg     0.8654    0.8634    0.8626       593

```