# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz16_seed42/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7631 |
| Accuracy | 0.7386 |
| Macro-AUROC (OvR) | 0.9139 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7335 |
| Balanced Acc | 0.7908 |
| Macro-Sensitivity | 0.7908 |
| Macro-Specificity | 0.8671 |
| Macro-Precision | 0.7753 |
| AUC-PR (Macro) | 0.8667 |
| Cohen's Kappa | 0.6022 |
| MCC | 0.6258 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2296.9 (0.44 ms/张) |
| End-to-End FPS | 132.6 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8680 | 0.7172 | 0.6042 | 0.7125 | 0.9161 | 0.8775 | 0.7765 | 197 |
| normal | 0.5543 | 0.9243 | 0.8644 | 0.6755 | 0.7043 | 0.8742 | 0.8538 | 276 |
| sever | 0.9500 | 0.9598 | 0.8571 | 0.9012 | 0.9870 | 0.9900 | 0.9699 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=171 FP=112 TN=284 FN=26
- **normal**: TP=153 FP=24 TN=293 FN=123
- **sever**: TP=114 FP=19 TN=454 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 171 | 22 | 4 |
| normal | 108 | 153 | 15 |
| sever | 4 | 2 | 114 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6042    0.8680    0.7125       197
      normal     0.8644    0.5543    0.6755       276
       sever     0.8571    0.9500    0.9012       120

    accuracy                         0.7386       593
   macro avg     0.7753    0.7908    0.7631       593
weighted avg     0.7765    0.7386    0.7335       593

```