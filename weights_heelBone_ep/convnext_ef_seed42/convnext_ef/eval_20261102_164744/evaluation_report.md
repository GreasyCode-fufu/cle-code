# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_ef_seed42/convnext_ef/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8751 |
| Accuracy | 0.8718 |
| Macro-AUROC (OvR) | 0.9745 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8712 |
| Balanced Acc | 0.8921 |
| Macro-Sensitivity | 0.8921 |
| Macro-Specificity | 0.9346 |
| Macro-Precision | 0.8641 |
| AUC-PR (Macro) | 0.9647 |
| Cohen's Kappa | 0.8005 |
| MCC | 0.8024 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 596.4 (1.68 ms/张) |
| End-to-End FPS | 239.2 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8477 | 0.9293 | 0.8564 | 0.8520 | 0.9246 | 0.9606 | 0.9457 | 197 |
| normal | 0.8370 | 0.9274 | 0.9094 | 0.8717 | 0.8673 | 0.9649 | 0.9563 | 276 |
| sever | 0.9917 | 0.9471 | 0.8264 | 0.9015 | 0.9978 | 0.9979 | 0.9919 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=167 FP=28 TN=368 FN=30
- **normal**: TP=231 FP=23 TN=294 FN=45
- **sever**: TP=119 FP=25 TN=448 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 167 | 22 | 8 |
| normal | 28 | 231 | 17 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8564    0.8477    0.8520       197
      normal     0.9094    0.8370    0.8717       276
       sever     0.8264    0.9917    0.9015       120

    accuracy                         0.8718       593
   macro avg     0.8641    0.8921    0.8751       593
weighted avg     0.8750    0.8718    0.8712       593

```