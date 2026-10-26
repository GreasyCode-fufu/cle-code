# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/convnext_v2bc_seed42/convnext_v2bc/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8588 |
| Accuracy | 0.8432 |
| Macro-AUROC (OvR) | 0.9795 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8413 |
| Balanced Acc | 0.8808 |
| Macro-Sensitivity | 0.8808 |
| Macro-Specificity | 0.9214 |
| Macro-Precision | 0.8598 |
| AUC-PR (Macro) | 0.9704 |
| Cohen's Kappa | 0.7590 |
| MCC | 0.7736 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.67M |
| Pure-Inference FPS | 676.4 (1.48 ms/张) |
| End-to-End FPS | 246.1 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9442 | 0.8232 | 0.7266 | 0.8212 | 0.9674 | 0.9694 | 0.9502 | 197 |
| normal | 0.7065 | 0.9685 | 0.9512 | 0.8108 | 0.7912 | 0.9701 | 0.9645 | 276 |
| sever | 0.9917 | 0.9725 | 0.9015 | 0.9444 | 0.9978 | 0.9991 | 0.9966 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=186 FP=70 TN=326 FN=11
- **normal**: TP=195 FP=10 TN=307 FN=81
- **sever**: TP=119 FP=13 TN=460 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 186 | 9 | 2 |
| normal | 70 | 195 | 11 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7266    0.9442    0.8212       197
      normal     0.9512    0.7065    0.8108       276
       sever     0.9015    0.9917    0.9444       120

    accuracy                         0.8432       593
   macro avg     0.8598    0.8808    0.8588       593
weighted avg     0.8665    0.8432    0.8413       593

```