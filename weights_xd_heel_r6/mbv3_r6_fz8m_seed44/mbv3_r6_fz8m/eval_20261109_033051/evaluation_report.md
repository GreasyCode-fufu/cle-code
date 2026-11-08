# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz8m_seed44/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8421 |
| Accuracy | 0.8297 |
| Macro-AUROC (OvR) | 0.9523 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8293 |
| Balanced Acc | 0.8589 |
| Macro-Sensitivity | 0.8589 |
| Macro-Specificity | 0.9124 |
| Macro-Precision | 0.8335 |
| AUC-PR (Macro) | 0.9263 |
| Cohen's Kappa | 0.7359 |
| MCC | 0.7404 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4881.0 (0.20 ms/张) |
| End-to-End FPS | 291.0 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8325 | 0.8510 | 0.7354 | 0.7810 | 0.9108 | 0.9265 | 0.8701 | 197 |
| normal | 0.7609 | 0.9243 | 0.8974 | 0.8235 | 0.8162 | 0.9334 | 0.9200 | 276 |
| sever | 0.9833 | 0.9619 | 0.8676 | 0.9219 | 0.9956 | 0.9971 | 0.9888 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=164 FP=59 TN=337 FN=33
- **normal**: TP=210 FP=24 TN=293 FN=66
- **sever**: TP=118 FP=18 TN=455 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 164 | 24 | 9 |
| normal | 57 | 210 | 9 |
| sever | 2 | 0 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7354    0.8325    0.7810       197
      normal     0.8974    0.7609    0.8235       276
       sever     0.8676    0.9833    0.9219       120

    accuracy                         0.8297       593
   macro avg     0.8335    0.8589    0.8421       593
weighted avg     0.8376    0.8297    0.8293       593

```