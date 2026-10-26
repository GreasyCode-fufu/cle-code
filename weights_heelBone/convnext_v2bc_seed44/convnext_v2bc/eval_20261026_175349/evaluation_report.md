# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/convnext_v2bc_seed44/convnext_v2bc/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8825 |
| Accuracy | 0.8769 |
| Macro-AUROC (OvR) | 0.9779 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8760 |
| Balanced Acc | 0.8963 |
| Macro-Sensitivity | 0.8963 |
| Macro-Specificity | 0.9359 |
| Macro-Precision | 0.8726 |
| AUC-PR (Macro) | 0.9681 |
| Cohen's Kappa | 0.8076 |
| MCC | 0.8088 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.67M |
| Pure-Inference FPS | 517.9 (1.93 ms/张) |
| End-to-End FPS | 224.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8376 | 0.9268 | 0.8505 | 0.8440 | 0.9198 | 0.9660 | 0.9494 | 197 |
| normal | 0.8514 | 0.9211 | 0.9038 | 0.8769 | 0.8769 | 0.9691 | 0.9611 | 276 |
| sever | 1.0000 | 0.9598 | 0.8633 | 0.9266 | 1.0000 | 0.9985 | 0.9939 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=165 FP=29 TN=367 FN=32
- **normal**: TP=235 FP=25 TN=292 FN=41
- **sever**: TP=120 FP=19 TN=454 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 165 | 25 | 7 |
| normal | 29 | 235 | 12 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8505    0.8376    0.8440       197
      normal     0.9038    0.8514    0.8769       276
       sever     0.8633    1.0000    0.9266       120

    accuracy                         0.8769       593
   macro avg     0.8726    0.8963    0.8825       593
weighted avg     0.8779    0.8769    0.8760       593

```