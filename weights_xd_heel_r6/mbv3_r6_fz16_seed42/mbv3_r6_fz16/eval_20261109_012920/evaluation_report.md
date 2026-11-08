# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz16_seed42/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7572 |
| Accuracy | 0.7386 |
| Macro-AUROC (OvR) | 0.9135 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7323 |
| Balanced Acc | 0.7908 |
| Macro-Sensitivity | 0.7908 |
| Macro-Specificity | 0.8690 |
| Macro-Precision | 0.7665 |
| AUC-PR (Macro) | 0.8670 |
| Cohen's Kappa | 0.6038 |
| MCC | 0.6272 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2853.0 (0.35 ms/张) |
| End-to-End FPS | 147.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8680 | 0.7323 | 0.6173 | 0.7215 | 0.9177 | 0.8744 | 0.7749 | 197 |
| normal | 0.5543 | 0.9338 | 0.8793 | 0.6800 | 0.7064 | 0.8745 | 0.8561 | 276 |
| sever | 0.9500 | 0.9408 | 0.8028 | 0.8702 | 0.9867 | 0.9917 | 0.9699 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=171 FP=106 TN=290 FN=26
- **normal**: TP=153 FP=21 TN=296 FN=123
- **sever**: TP=114 FP=28 TN=445 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 171 | 19 | 7 |
| normal | 102 | 153 | 21 |
| sever | 4 | 2 | 114 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6173    0.8680    0.7215       197
      normal     0.8793    0.5543    0.6800       276
       sever     0.8028    0.9500    0.8702       120

    accuracy                         0.7386       593
   macro avg     0.7665    0.7908    0.7572       593
weighted avg     0.7768    0.7386    0.7323       593

```