# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_ef_seed44/convnext_ef/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8852 |
| Accuracy | 0.8735 |
| Macro-AUROC (OvR) | 0.9792 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8730 |
| Balanced Acc | 0.9017 |
| Macro-Sensitivity | 0.9017 |
| Macro-Specificity | 0.9360 |
| Macro-Precision | 0.8790 |
| AUC-PR (Macro) | 0.9697 |
| Cohen's Kappa | 0.8040 |
| MCC | 0.8104 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 1126.1 (0.89 ms/张) |
| End-to-End FPS | 278.7 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9188 | 0.8712 | 0.7802 | 0.8438 | 0.9557 | 0.9689 | 0.9468 | 197 |
| normal | 0.7862 | 0.9621 | 0.9476 | 0.8594 | 0.8379 | 0.9696 | 0.9653 | 276 |
| sever | 1.0000 | 0.9746 | 0.9091 | 0.9524 | 1.0000 | 0.9992 | 0.9969 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=181 FP=51 TN=345 FN=16
- **normal**: TP=217 FP=12 TN=305 FN=59
- **sever**: TP=120 FP=12 TN=461 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 181 | 12 | 4 |
| normal | 51 | 217 | 8 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7802    0.9188    0.8438       197
      normal     0.9476    0.7862    0.8594       276
       sever     0.9091    1.0000    0.9524       120

    accuracy                         0.8735       593
   macro avg     0.8790    0.9017    0.8852       593
weighted avg     0.8842    0.8735    0.8730       593

```