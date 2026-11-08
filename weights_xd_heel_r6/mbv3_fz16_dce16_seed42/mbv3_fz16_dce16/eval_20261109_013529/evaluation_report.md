# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce16_seed42/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8237 |
| Accuracy | 0.8078 |
| Macro-AUROC (OvR) | 0.9335 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8068 |
| Balanced Acc | 0.8258 |
| Macro-Sensitivity | 0.8258 |
| Macro-Specificity | 0.8935 |
| Macro-Precision | 0.8222 |
| AUC-PR (Macro) | 0.8976 |
| Cohen's Kappa | 0.6957 |
| MCC | 0.6959 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 3695.8 (0.27 ms/张) |
| End-to-End FPS | 18.3 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7157 | 0.8813 | 0.7500 | 0.7325 | 0.8617 | 0.9023 | 0.8229 | 197 |
| normal | 0.8116 | 0.8202 | 0.7972 | 0.8043 | 0.8333 | 0.9018 | 0.8843 | 276 |
| sever | 0.9500 | 0.9789 | 0.9194 | 0.9344 | 0.9872 | 0.9963 | 0.9856 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=141 FP=47 TN=349 FN=56
- **normal**: TP=224 FP=57 TN=260 FN=52
- **sever**: TP=114 FP=10 TN=463 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 141 | 53 | 3 |
| normal | 45 | 224 | 7 |
| sever | 2 | 4 | 114 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7500    0.7157    0.7325       197
      normal     0.7972    0.8116    0.8043       276
       sever     0.9194    0.9500    0.9344       120

    accuracy                         0.8078       593
   macro avg     0.8222    0.8258    0.8237       593
weighted avg     0.8062    0.8078    0.8068       593

```