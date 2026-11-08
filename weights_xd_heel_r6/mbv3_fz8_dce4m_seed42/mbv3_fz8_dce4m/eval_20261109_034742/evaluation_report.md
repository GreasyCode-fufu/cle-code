# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz8_dce4m_seed42/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8473 |
| Accuracy | 0.8347 |
| Macro-AUROC (OvR) | 0.9512 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8336 |
| Balanced Acc | 0.8561 |
| Macro-Sensitivity | 0.8561 |
| Macro-Specificity | 0.9103 |
| Macro-Precision | 0.8400 |
| AUC-PR (Macro) | 0.9292 |
| Cohen's Kappa | 0.7404 |
| MCC | 0.7407 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4161.3 (0.24 ms/张) |
| End-to-End FPS | 13.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7817 | 0.9015 | 0.7979 | 0.7897 | 0.8925 | 0.9310 | 0.8856 | 197 |
| normal | 0.8116 | 0.8612 | 0.8358 | 0.8235 | 0.8400 | 0.9256 | 0.9134 | 276 |
| sever | 0.9750 | 0.9683 | 0.8864 | 0.9286 | 0.9935 | 0.9970 | 0.9885 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=154 FP=39 TN=357 FN=43
- **normal**: TP=224 FP=44 TN=273 FN=52
- **sever**: TP=117 FP=15 TN=458 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 154 | 41 | 2 |
| normal | 39 | 224 | 13 |
| sever | 0 | 3 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7979    0.7817    0.7897       197
      normal     0.8358    0.8116    0.8235       276
       sever     0.8864    0.9750    0.9286       120

    accuracy                         0.8347       593
   macro avg     0.8400    0.8561    0.8473       593
weighted avg     0.8335    0.8347    0.8336       593

```