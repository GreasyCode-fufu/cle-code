# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce16_seed42/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8569 |
| Accuracy | 0.8415 |
| Macro-AUROC (OvR) | 0.9459 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8417 |
| Balanced Acc | 0.8542 |
| Macro-Sensitivity | 0.8542 |
| Macro-Specificity | 0.9114 |
| Macro-Precision | 0.8599 |
| AUC-PR (Macro) | 0.9189 |
| Cohen's Kappa | 0.7485 |
| MCC | 0.7486 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 4219.1 (0.24 ms/张) |
| End-to-End FPS | 20.1 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7766 | 0.8939 | 0.7846 | 0.7806 | 0.8894 | 0.9221 | 0.8678 | 197 |
| normal | 0.8442 | 0.8486 | 0.8292 | 0.8366 | 0.8622 | 0.9196 | 0.9034 | 276 |
| sever | 0.9417 | 0.9915 | 0.9658 | 0.9536 | 0.9853 | 0.9961 | 0.9855 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=153 FP=42 TN=354 FN=44
- **normal**: TP=233 FP=48 TN=269 FN=43
- **sever**: TP=113 FP=4 TN=469 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 153 | 44 | 0 |
| normal | 39 | 233 | 4 |
| sever | 3 | 4 | 113 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7846    0.7766    0.7806       197
      normal     0.8292    0.8442    0.8366       276
       sever     0.9658    0.9417    0.9536       120

    accuracy                         0.8415       593
   macro avg     0.8599    0.8542    0.8569       593
weighted avg     0.8420    0.8415    0.8417       593

```