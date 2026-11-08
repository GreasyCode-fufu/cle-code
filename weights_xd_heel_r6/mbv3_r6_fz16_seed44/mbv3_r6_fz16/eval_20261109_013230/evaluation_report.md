# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz16_seed44/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7749 |
| Accuracy | 0.7555 |
| Macro-AUROC (OvR) | 0.9227 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7512 |
| Balanced Acc | 0.8041 |
| Macro-Sensitivity | 0.8041 |
| Macro-Specificity | 0.8761 |
| Macro-Precision | 0.7762 |
| AUC-PR (Macro) | 0.8792 |
| Cohen's Kappa | 0.6272 |
| MCC | 0.6441 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3538.7 (0.28 ms/张) |
| End-to-End FPS | 313.3 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8477 | 0.7601 | 0.6374 | 0.7277 | 0.9094 | 0.8908 | 0.8015 | 197 |
| normal | 0.5978 | 0.9211 | 0.8684 | 0.7082 | 0.7246 | 0.8864 | 0.8682 | 276 |
| sever | 0.9667 | 0.9471 | 0.8227 | 0.8889 | 0.9912 | 0.9908 | 0.9678 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=167 FP=95 TN=301 FN=30
- **normal**: TP=165 FP=25 TN=292 FN=111
- **sever**: TP=116 FP=25 TN=448 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 167 | 23 | 7 |
| normal | 93 | 165 | 18 |
| sever | 2 | 2 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6374    0.8477    0.7277       197
      normal     0.8684    0.5978    0.7082       276
       sever     0.8227    0.9667    0.8889       120

    accuracy                         0.7555       593
   macro avg     0.7762    0.8041    0.7749       593
weighted avg     0.7824    0.7555    0.7512       593

```