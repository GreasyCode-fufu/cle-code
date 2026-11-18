# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/convnext_v2bc_seed43/convnext_v2bc/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8807 |
| Accuracy | 0.8769 |
| Macro-AUROC (OvR) | 0.9782 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8764 |
| Balanced Acc | 0.8982 |
| Macro-Sensitivity | 0.8982 |
| Macro-Specificity | 0.9379 |
| Macro-Precision | 0.8694 |
| AUC-PR (Macro) | 0.9681 |
| Cohen's Kappa | 0.8087 |
| MCC | 0.8112 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.67M |
| Pure-Inference FPS | 730.7 (1.37 ms/张) |
| End-to-End FPS | 258.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8731 | 0.9192 | 0.8431 | 0.8579 | 0.9357 | 0.9669 | 0.9462 | 197 |
| normal | 0.8297 | 0.9432 | 0.9271 | 0.8757 | 0.8642 | 0.9701 | 0.9655 | 276 |
| sever | 0.9917 | 0.9514 | 0.8380 | 0.9084 | 0.9978 | 0.9977 | 0.9925 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=172 FP=32 TN=364 FN=25
- **normal**: TP=229 FP=18 TN=299 FN=47
- **sever**: TP=119 FP=23 TN=450 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 172 | 17 | 8 |
| normal | 32 | 229 | 15 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8431    0.8731    0.8579       197
      normal     0.9271    0.8297    0.8757       276
       sever     0.8380    0.9917    0.9084       120

    accuracy                         0.8769       593
   macro avg     0.8694    0.8982    0.8807       593
weighted avg     0.8812    0.8769    0.8764       593

```