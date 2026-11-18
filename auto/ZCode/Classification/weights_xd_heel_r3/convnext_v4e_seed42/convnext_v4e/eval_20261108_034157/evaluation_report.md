# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r3/convnext_v4e_seed42/convnext_v4e/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8966 |
| Accuracy | 0.8887 |
| Macro-AUROC (OvR) | 0.9770 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8879 |
| Balanced Acc | 0.9023 |
| Macro-Sensitivity | 0.9023 |
| Macro-Specificity | 0.9395 |
| Macro-Precision | 0.8919 |
| AUC-PR (Macro) | 0.9668 |
| Cohen's Kappa | 0.8245 |
| MCC | 0.8248 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 428.9 (2.33 ms/张) |
| End-to-End FPS | 208.1 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8274 | 0.9343 | 0.8624 | 0.8446 | 0.9158 | 0.9635 | 0.9421 | 197 |
| normal | 0.8877 | 0.9054 | 0.8909 | 0.8893 | 0.9025 | 0.9685 | 0.9613 | 276 |
| sever | 0.9917 | 0.9789 | 0.9225 | 0.9558 | 0.9978 | 0.9990 | 0.9970 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=163 FP=26 TN=370 FN=34
- **normal**: TP=245 FP=30 TN=287 FN=31
- **sever**: TP=119 FP=10 TN=463 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 163 | 29 | 5 |
| normal | 26 | 245 | 5 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8624    0.8274    0.8446       197
      normal     0.8909    0.8877    0.8893       276
       sever     0.9225    0.9917    0.9558       120

    accuracy                         0.8887       593
   macro avg     0.8919    0.9023    0.8966       593
weighted avg     0.8878    0.8887    0.8879       593

```