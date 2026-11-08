# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz16bn_seed43/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7893 |
| Accuracy | 0.7757 |
| Macro-AUROC (OvR) | 0.9272 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7737 |
| Balanced Acc | 0.8158 |
| Macro-Sensitivity | 0.8158 |
| Macro-Specificity | 0.8863 |
| Macro-Precision | 0.7809 |
| AUC-PR (Macro) | 0.8850 |
| Cohen's Kappa | 0.6561 |
| MCC | 0.6649 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3601.2 (0.28 ms/张) |
| End-to-End FPS | 117.3 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8020 | 0.8106 | 0.6781 | 0.7349 | 0.8917 | 0.8870 | 0.7937 | 197 |
| normal | 0.6703 | 0.9117 | 0.8685 | 0.7566 | 0.7605 | 0.9006 | 0.8815 | 276 |
| sever | 0.9750 | 0.9366 | 0.7959 | 0.8764 | 0.9933 | 0.9941 | 0.9799 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=158 FP=75 TN=321 FN=39
- **normal**: TP=185 FP=28 TN=289 FN=91
- **sever**: TP=117 FP=30 TN=443 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 158 | 26 | 13 |
| normal | 74 | 185 | 17 |
| sever | 1 | 2 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6781    0.8020    0.7349       197
      normal     0.8685    0.6703    0.7566       276
       sever     0.7959    0.9750    0.8764       120

    accuracy                         0.7757       593
   macro avg     0.7809    0.8158    0.7893       593
weighted avg     0.7906    0.7757    0.7737       593

```