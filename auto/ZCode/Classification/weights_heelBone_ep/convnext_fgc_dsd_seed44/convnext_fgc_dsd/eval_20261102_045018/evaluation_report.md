# convnext_fgc_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_fgc_dsd_seed44/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8594 |
| Accuracy | 0.8634 |
| Macro-AUROC (OvR) | 0.9682 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8636 |
| Balanced Acc | 0.8766 |
| Macro-Sensitivity | 0.8766 |
| Macro-Specificity | 0.9307 |
| Macro-Precision | 0.8565 |
| AUC-PR (Macro) | 0.9532 |
| Cohen's Kappa | 0.7869 |
| MCC | 0.7910 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 953.4 (1.05 ms/张) |
| End-to-End FPS | 284.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7817 | 0.9722 | 0.9333 | 0.8508 | 0.8995 | 0.9515 | 0.9320 | 197 |
| normal | 0.8732 | 0.9022 | 0.8860 | 0.8796 | 0.8910 | 0.9616 | 0.9548 | 276 |
| sever | 0.9750 | 0.9175 | 0.7500 | 0.8478 | 0.9931 | 0.9915 | 0.9728 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=154 FP=11 TN=385 FN=43
- **normal**: TP=241 FP=31 TN=286 FN=35
- **sever**: TP=117 FP=39 TN=434 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 154 | 28 | 15 |
| normal | 11 | 241 | 24 |
| sever | 0 | 3 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.9333    0.7817    0.8508       197
      normal     0.8860    0.8732    0.8796       276
       sever     0.7500    0.9750    0.8478       120

    accuracy                         0.8634       593
   macro avg     0.8565    0.8766    0.8594       593
weighted avg     0.8742    0.8634    0.8636       593

```