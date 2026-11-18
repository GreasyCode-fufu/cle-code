# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/resnet50_seed43/resnet50/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8941 |
| Accuracy | 0.8887 |
| Macro-AUROC (OvR) | 0.9749 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8878 |
| Balanced Acc | 0.8940 |
| Macro-Sensitivity | 0.8940 |
| Macro-Specificity | 0.9376 |
| Macro-Precision | 0.8966 |
| AUC-PR (Macro) | 0.9618 |
| Cohen's Kappa | 0.8232 |
| MCC | 0.8246 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 839.4 (1.19 ms/张) |
| End-to-End FPS | 262.7 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8071 | 0.9571 | 0.9034 | 0.8525 | 0.9089 | 0.9661 | 0.9456 | 197 |
| normal | 0.9167 | 0.8770 | 0.8664 | 0.8908 | 0.9236 | 0.9643 | 0.9532 | 276 |
| sever | 0.9583 | 0.9789 | 0.9200 | 0.9388 | 0.9893 | 0.9943 | 0.9866 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=159 FP=17 TN=379 FN=38
- **normal**: TP=253 FP=39 TN=278 FN=23
- **sever**: TP=115 FP=10 TN=463 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 159 | 34 | 4 |
| normal | 17 | 253 | 6 |
| sever | 0 | 5 | 115 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.9034    0.8071    0.8525       197
      normal     0.8664    0.9167    0.8908       276
       sever     0.9200    0.9583    0.9388       120

    accuracy                         0.8887       593
   macro avg     0.8966    0.8940    0.8941       593
weighted avg     0.8896    0.8887    0.8878       593

```