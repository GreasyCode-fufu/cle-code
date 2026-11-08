# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce4m_seed42/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8355 |
| Accuracy | 0.8196 |
| Macro-AUROC (OvR) | 0.9346 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8187 |
| Balanced Acc | 0.8377 |
| Macro-Sensitivity | 0.8377 |
| Macro-Specificity | 0.9000 |
| Macro-Precision | 0.8335 |
| AUC-PR (Macro) | 0.9014 |
| Cohen's Kappa | 0.7146 |
| MCC | 0.7147 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3829.7 (0.26 ms/张) |
| End-to-End FPS | 18.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7360 | 0.8864 | 0.7632 | 0.7494 | 0.8710 | 0.9036 | 0.8332 | 197 |
| normal | 0.8188 | 0.8328 | 0.8100 | 0.8144 | 0.8408 | 0.9035 | 0.8833 | 276 |
| sever | 0.9583 | 0.9810 | 0.9274 | 0.9426 | 0.9893 | 0.9968 | 0.9877 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=145 FP=45 TN=351 FN=52
- **normal**: TP=226 FP=53 TN=264 FN=50
- **sever**: TP=115 FP=9 TN=464 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 145 | 49 | 3 |
| normal | 44 | 226 | 6 |
| sever | 1 | 4 | 115 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7632    0.7360    0.7494       197
      normal     0.8100    0.8188    0.8144       276
       sever     0.9274    0.9583    0.9426       120

    accuracy                         0.8196       593
   macro avg     0.8335    0.8377    0.8355       593
weighted avg     0.8182    0.8196    0.8187       593

```