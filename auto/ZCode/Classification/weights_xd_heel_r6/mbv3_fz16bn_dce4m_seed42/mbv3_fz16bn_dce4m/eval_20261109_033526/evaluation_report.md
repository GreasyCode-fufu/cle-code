# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16bn_dce4m_seed42/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8497 |
| Accuracy | 0.8314 |
| Macro-AUROC (OvR) | 0.9383 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8310 |
| Balanced Acc | 0.8522 |
| Macro-Sensitivity | 0.8522 |
| Macro-Specificity | 0.9066 |
| Macro-Precision | 0.8473 |
| AUC-PR (Macro) | 0.9056 |
| Cohen's Kappa | 0.7338 |
| MCC | 0.7338 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4217.1 (0.24 ms/张) |
| End-to-End FPS | 11.3 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7665 | 0.8838 | 0.7665 | 0.7665 | 0.8838 | 0.9083 | 0.8416 | 197 |
| normal | 0.8152 | 0.8486 | 0.8242 | 0.8197 | 0.8406 | 0.9084 | 0.8813 | 276 |
| sever | 0.9750 | 0.9873 | 0.9512 | 0.9630 | 0.9936 | 0.9982 | 0.9941 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=151 FP=46 TN=350 FN=46
- **normal**: TP=225 FP=48 TN=269 FN=51
- **sever**: TP=117 FP=6 TN=467 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 151 | 45 | 1 |
| normal | 46 | 225 | 5 |
| sever | 0 | 3 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7665    0.7665    0.7665       197
      normal     0.8242    0.8152    0.8197       276
       sever     0.9512    0.9750    0.9630       120

    accuracy                         0.8314       593
   macro avg     0.8473    0.8522    0.8497       593
weighted avg     0.8307    0.8314    0.8310       593

```