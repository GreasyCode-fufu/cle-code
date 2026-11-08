# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz8m_seed44/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8115 |
| Accuracy | 0.7943 |
| Macro-AUROC (OvR) | 0.9414 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7918 |
| Balanced Acc | 0.8364 |
| Macro-Sensitivity | 0.8364 |
| Macro-Specificity | 0.8956 |
| Macro-Precision | 0.8081 |
| AUC-PR (Macro) | 0.9140 |
| Cohen's Kappa | 0.6845 |
| MCC | 0.6969 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4859.9 (0.21 ms/张) |
| End-to-End FPS | 236.2 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8629 | 0.8005 | 0.6827 | 0.7623 | 0.9215 | 0.9146 | 0.8513 | 197 |
| normal | 0.6630 | 0.9306 | 0.8927 | 0.7609 | 0.7603 | 0.9131 | 0.9038 | 276 |
| sever | 0.9833 | 0.9556 | 0.8489 | 0.9112 | 0.9956 | 0.9964 | 0.9869 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=170 FP=79 TN=317 FN=27
- **normal**: TP=183 FP=22 TN=295 FN=93
- **sever**: TP=118 FP=21 TN=452 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 170 | 21 | 6 |
| normal | 78 | 183 | 15 |
| sever | 1 | 1 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6827    0.8629    0.7623       197
      normal     0.8927    0.6630    0.7609       276
       sever     0.8489    0.9833    0.9112       120

    accuracy                         0.7943       593
   macro avg     0.8081    0.8364    0.8115       593
weighted avg     0.8141    0.7943    0.7918       593

```