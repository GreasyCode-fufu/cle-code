# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/convnext_v2bc_seed42/convnext_v2bc/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8670 |
| Accuracy | 0.8499 |
| Macro-AUROC (OvR) | 0.9774 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8494 |
| Balanced Acc | 0.8837 |
| Macro-Sensitivity | 0.8837 |
| Macro-Specificity | 0.9236 |
| Macro-Precision | 0.8668 |
| AUC-PR (Macro) | 0.9675 |
| Cohen's Kappa | 0.7681 |
| MCC | 0.7791 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.67M |
| Pure-Inference FPS | 522.5 (1.91 ms/张) |
| End-to-End FPS | 228.7 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9239 | 0.8308 | 0.7309 | 0.8161 | 0.9564 | 0.9664 | 0.9456 | 197 |
| normal | 0.7355 | 0.9590 | 0.9398 | 0.8252 | 0.8064 | 0.9669 | 0.9615 | 276 |
| sever | 0.9917 | 0.9810 | 0.9297 | 0.9597 | 0.9978 | 0.9988 | 0.9953 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=182 FP=67 TN=329 FN=15
- **normal**: TP=203 FP=13 TN=304 FN=73
- **sever**: TP=119 FP=9 TN=464 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 182 | 12 | 3 |
| normal | 67 | 203 | 6 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7309    0.9239    0.8161       197
      normal     0.9398    0.7355    0.8252       276
       sever     0.9297    0.9917    0.9597       120

    accuracy                         0.8499       593
   macro avg     0.8668    0.8837    0.8670       593
weighted avg     0.8684    0.8499    0.8494       593

```