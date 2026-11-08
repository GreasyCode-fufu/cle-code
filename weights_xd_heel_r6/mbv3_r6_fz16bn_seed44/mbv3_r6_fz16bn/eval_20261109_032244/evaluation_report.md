# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz16bn_seed44/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7770 |
| Accuracy | 0.7572 |
| Macro-AUROC (OvR) | 0.9172 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7540 |
| Balanced Acc | 0.8004 |
| Macro-Sensitivity | 0.8004 |
| Macro-Specificity | 0.8739 |
| Macro-Precision | 0.7689 |
| AUC-PR (Macro) | 0.8694 |
| Cohen's Kappa | 0.6265 |
| MCC | 0.6346 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3936.3 (0.25 ms/张) |
| End-to-End FPS | 155.1 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7970 | 0.7955 | 0.6597 | 0.7218 | 0.8873 | 0.8823 | 0.7868 | 197 |
| normal | 0.6377 | 0.8770 | 0.8186 | 0.7169 | 0.7354 | 0.8749 | 0.8398 | 276 |
| sever | 0.9667 | 0.9493 | 0.8286 | 0.8923 | 0.9912 | 0.9945 | 0.9817 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=157 FP=81 TN=315 FN=40
- **normal**: TP=176 FP=39 TN=278 FN=100
- **sever**: TP=116 FP=24 TN=449 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 157 | 35 | 5 |
| normal | 81 | 176 | 19 |
| sever | 0 | 4 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6597    0.7970    0.7218       197
      normal     0.8186    0.6377    0.7169       276
       sever     0.8286    0.9667    0.8923       120

    accuracy                         0.7572       593
   macro avg     0.7689    0.8004    0.7770       593
weighted avg     0.7678    0.7572    0.7540       593

```