# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_ef_seed43/convnext_ef/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8666 |
| Accuracy | 0.8499 |
| Macro-AUROC (OvR) | 0.9716 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8498 |
| Balanced Acc | 0.8773 |
| Macro-Sensitivity | 0.8773 |
| Macro-Specificity | 0.9209 |
| Macro-Precision | 0.8621 |
| AUC-PR (Macro) | 0.9604 |
| Cohen's Kappa | 0.7661 |
| MCC | 0.7702 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 597.1 (1.67 ms/张) |
| End-to-End FPS | 243.1 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8731 | 0.8586 | 0.7544 | 0.8094 | 0.9315 | 0.9568 | 0.9349 | 197 |
| normal | 0.7754 | 0.9211 | 0.8954 | 0.8311 | 0.8249 | 0.9584 | 0.9483 | 276 |
| sever | 0.9833 | 0.9831 | 0.9365 | 0.9593 | 0.9957 | 0.9995 | 0.9980 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=172 FP=56 TN=340 FN=25
- **normal**: TP=214 FP=25 TN=292 FN=62
- **sever**: TP=118 FP=8 TN=465 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 172 | 24 | 1 |
| normal | 55 | 214 | 7 |
| sever | 1 | 1 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7544    0.8731    0.8094       197
      normal     0.8954    0.7754    0.8311       276
       sever     0.9365    0.9833    0.9593       120

    accuracy                         0.8499       593
   macro avg     0.8621    0.8773    0.8666       593
weighted avg     0.8569    0.8499    0.8498       593

```