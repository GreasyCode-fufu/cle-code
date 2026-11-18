# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce4m_seed43/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8476 |
| Accuracy | 0.8331 |
| Macro-AUROC (OvR) | 0.9412 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8321 |
| Balanced Acc | 0.8495 |
| Macro-Sensitivity | 0.8495 |
| Macro-Specificity | 0.9073 |
| Macro-Precision | 0.8463 |
| AUC-PR (Macro) | 0.9083 |
| Cohen's Kappa | 0.7357 |
| MCC | 0.7359 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3843.5 (0.26 ms/张) |
| End-to-End FPS | 18.6 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7411 | 0.8965 | 0.7807 | 0.7604 | 0.8744 | 0.9130 | 0.8489 | 197 |
| normal | 0.8406 | 0.8423 | 0.8227 | 0.8315 | 0.8585 | 0.9140 | 0.8904 | 276 |
| sever | 0.9667 | 0.9831 | 0.9355 | 0.9508 | 0.9915 | 0.9965 | 0.9857 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=146 FP=41 TN=355 FN=51
- **normal**: TP=232 FP=50 TN=267 FN=44
- **sever**: TP=116 FP=8 TN=465 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 146 | 48 | 3 |
| normal | 39 | 232 | 5 |
| sever | 2 | 2 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7807    0.7411    0.7604       197
      normal     0.8227    0.8406    0.8315       276
       sever     0.9355    0.9667    0.9508       120

    accuracy                         0.8331       593
   macro avg     0.8463    0.8495    0.8476       593
weighted avg     0.8316    0.8331    0.8321       593

```