# convnext_fgc_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_fgc_dsd_seed42/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8757 |
| Accuracy | 0.8735 |
| Macro-AUROC (OvR) | 0.9737 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8727 |
| Balanced Acc | 0.8884 |
| Macro-Sensitivity | 0.8884 |
| Macro-Specificity | 0.9339 |
| Macro-Precision | 0.8678 |
| AUC-PR (Macro) | 0.9616 |
| Cohen's Kappa | 0.8018 |
| MCC | 0.8031 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 1015.8 (0.98 ms/张) |
| End-to-End FPS | 287.0 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8122 | 0.9419 | 0.8743 | 0.8421 | 0.9098 | 0.9577 | 0.9349 | 197 |
| normal | 0.8696 | 0.9085 | 0.8922 | 0.8807 | 0.8889 | 0.9662 | 0.9596 | 276 |
| sever | 0.9833 | 0.9514 | 0.8369 | 0.9042 | 0.9956 | 0.9971 | 0.9902 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=160 FP=23 TN=373 FN=37
- **normal**: TP=240 FP=29 TN=288 FN=36
- **sever**: TP=118 FP=23 TN=450 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 160 | 28 | 9 |
| normal | 22 | 240 | 14 |
| sever | 1 | 1 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8743    0.8122    0.8421       197
      normal     0.8922    0.8696    0.8807       276
       sever     0.8369    0.9833    0.9042       120

    accuracy                         0.8735       593
   macro avg     0.8678    0.8884    0.8757       593
weighted avg     0.8751    0.8735    0.8727       593

```