# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz16bn_seed42/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7460 |
| Accuracy | 0.7218 |
| Macro-AUROC (OvR) | 0.9088 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7167 |
| Balanced Acc | 0.7760 |
| Macro-Sensitivity | 0.7760 |
| Macro-Specificity | 0.8580 |
| Macro-Precision | 0.7492 |
| AUC-PR (Macro) | 0.8599 |
| Cohen's Kappa | 0.5767 |
| MCC | 0.5947 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3690.8 (0.27 ms/张) |
| End-to-End FPS | 117.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8071 | 0.7247 | 0.5933 | 0.6839 | 0.8831 | 0.8663 | 0.7652 | 197 |
| normal | 0.5543 | 0.9022 | 0.8315 | 0.6652 | 0.6993 | 0.8688 | 0.8456 | 276 |
| sever | 0.9667 | 0.9471 | 0.8227 | 0.8889 | 0.9912 | 0.9915 | 0.9689 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=159 FP=109 TN=287 FN=38
- **normal**: TP=153 FP=31 TN=286 FN=123
- **sever**: TP=116 FP=25 TN=448 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 159 | 29 | 9 |
| normal | 107 | 153 | 16 |
| sever | 2 | 2 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.5933    0.8071    0.6839       197
      normal     0.8315    0.5543    0.6652       276
       sever     0.8227    0.9667    0.8889       120

    accuracy                         0.7218       593
   macro avg     0.7492    0.7760    0.7460       593
weighted avg     0.7506    0.7218    0.7167       593

```