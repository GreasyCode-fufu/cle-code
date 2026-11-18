# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r5/convnext_v6f_seed44/convnext_v6f/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8581 |
| Accuracy | 0.8583 |
| Macro-AUROC (OvR) | 0.9745 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8585 |
| Balanced Acc | 0.8821 |
| Macro-Sensitivity | 0.8821 |
| Macro-Specificity | 0.9297 |
| Macro-Precision | 0.8488 |
| AUC-PR (Macro) | 0.9639 |
| Cohen's Kappa | 0.7809 |
| MCC | 0.7855 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 1416.9 (0.71 ms/张) |
| End-to-End FPS | 312.0 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8274 | 0.9545 | 0.9006 | 0.8624 | 0.9175 | 0.9623 | 0.9443 | 197 |
| normal | 0.8188 | 0.9211 | 0.9004 | 0.8577 | 0.8538 | 0.9624 | 0.9513 | 276 |
| sever | 1.0000 | 0.9133 | 0.7453 | 0.8541 | 1.0000 | 0.9989 | 0.9961 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=163 FP=18 TN=378 FN=34
- **normal**: TP=226 FP=25 TN=292 FN=50
- **sever**: TP=120 FP=41 TN=432 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 163 | 25 | 9 |
| normal | 18 | 226 | 32 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.9006    0.8274    0.8624       197
      normal     0.9004    0.8188    0.8577       276
       sever     0.7453    1.0000    0.8541       120

    accuracy                         0.8583       593
   macro avg     0.8488    0.8821    0.8581       593
weighted avg     0.8691    0.8583    0.8585       593

```