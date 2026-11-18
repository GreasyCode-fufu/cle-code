# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz16bn_seed42/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7601 |
| Accuracy | 0.7386 |
| Macro-AUROC (OvR) | 0.9145 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7359 |
| Balanced Acc | 0.7855 |
| Macro-Sensitivity | 0.7855 |
| Macro-Specificity | 0.8667 |
| Macro-Precision | 0.7622 |
| AUC-PR (Macro) | 0.8650 |
| Cohen's Kappa | 0.6007 |
| MCC | 0.6161 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3645.3 (0.27 ms/张) |
| End-to-End FPS | 130.3 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8122 | 0.7424 | 0.6107 | 0.6972 | 0.8882 | 0.8698 | 0.7776 | 197 |
| normal | 0.5942 | 0.9085 | 0.8497 | 0.6994 | 0.7200 | 0.8840 | 0.8538 | 276 |
| sever | 0.9500 | 0.9493 | 0.8261 | 0.8837 | 0.9868 | 0.9897 | 0.9635 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=160 FP=102 TN=294 FN=37
- **normal**: TP=164 FP=29 TN=288 FN=112
- **sever**: TP=114 FP=24 TN=449 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 160 | 25 | 12 |
| normal | 100 | 164 | 12 |
| sever | 2 | 4 | 114 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6107    0.8122    0.6972       197
      normal     0.8497    0.5942    0.6994       276
       sever     0.8261    0.9500    0.8837       120

    accuracy                         0.7386       593
   macro avg     0.7622    0.7855    0.7601       593
weighted avg     0.7655    0.7386    0.7359       593

```