# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce16_seed44/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8442 |
| Accuracy | 0.8246 |
| Macro-AUROC (OvR) | 0.9427 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8247 |
| Balanced Acc | 0.8444 |
| Macro-Sensitivity | 0.8444 |
| Macro-Specificity | 0.9025 |
| Macro-Precision | 0.8441 |
| AUC-PR (Macro) | 0.9132 |
| Cohen's Kappa | 0.7226 |
| MCC | 0.7226 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 3364.8 (0.30 ms/张) |
| End-to-End FPS | 18.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7513 | 0.8737 | 0.7475 | 0.7494 | 0.8759 | 0.9177 | 0.8479 | 197 |
| normal | 0.8152 | 0.8423 | 0.8182 | 0.8167 | 0.8396 | 0.9140 | 0.9046 | 276 |
| sever | 0.9667 | 0.9915 | 0.9667 | 0.9667 | 0.9915 | 0.9963 | 0.9871 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=148 FP=50 TN=346 FN=49
- **normal**: TP=225 FP=50 TN=267 FN=51
- **sever**: TP=116 FP=4 TN=469 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 148 | 49 | 0 |
| normal | 47 | 225 | 4 |
| sever | 3 | 1 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7475    0.7513    0.7494       197
      normal     0.8182    0.8152    0.8167       276
       sever     0.9667    0.9667    0.9667       120

    accuracy                         0.8246       593
   macro avg     0.8441    0.8444    0.8442       593
weighted avg     0.8247    0.8246    0.8247       593

```