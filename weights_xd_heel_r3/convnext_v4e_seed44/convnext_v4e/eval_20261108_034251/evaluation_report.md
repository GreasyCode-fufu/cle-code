# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r3/convnext_v4e_seed44/convnext_v4e/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8697 |
| Accuracy | 0.8550 |
| Macro-AUROC (OvR) | 0.9738 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8554 |
| Balanced Acc | 0.8797 |
| Macro-Sensitivity | 0.8797 |
| Macro-Specificity | 0.9245 |
| Macro-Precision | 0.8675 |
| AUC-PR (Macro) | 0.9624 |
| Cohen's Kappa | 0.7742 |
| MCC | 0.7795 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 581.9 (1.72 ms/张) |
| End-to-End FPS | 235.6 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8934 | 0.8535 | 0.7521 | 0.8167 | 0.9415 | 0.9596 | 0.9335 | 197 |
| normal | 0.7790 | 0.9369 | 0.9149 | 0.8415 | 0.8296 | 0.9632 | 0.9583 | 276 |
| sever | 0.9667 | 0.9831 | 0.9355 | 0.9508 | 0.9915 | 0.9986 | 0.9953 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=176 FP=58 TN=338 FN=21
- **normal**: TP=215 FP=20 TN=297 FN=61
- **sever**: TP=116 FP=8 TN=465 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 176 | 18 | 3 |
| normal | 56 | 215 | 5 |
| sever | 2 | 2 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7521    0.8934    0.8167       197
      normal     0.9149    0.7790    0.8415       276
       sever     0.9355    0.9667    0.9508       120

    accuracy                         0.8550       593
   macro avg     0.8675    0.8797    0.8697       593
weighted avg     0.8650    0.8550    0.8554       593

```