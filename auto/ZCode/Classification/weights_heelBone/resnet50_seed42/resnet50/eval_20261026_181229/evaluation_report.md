# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/resnet50_seed42/resnet50/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8359 |
| Accuracy | 0.8432 |
| Macro-AUROC (OvR) | 0.9710 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8454 |
| Balanced Acc | 0.8629 |
| Macro-Sensitivity | 0.8629 |
| Macro-Specificity | 0.9248 |
| Macro-Precision | 0.8359 |
| AUC-PR (Macro) | 0.9565 |
| Cohen's Kappa | 0.7584 |
| MCC | 0.7669 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 867.8 (1.15 ms/张) |
| End-to-End FPS | 274.6 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7563 | 0.9697 | 0.9255 | 0.8324 | 0.8889 | 0.9517 | 0.9306 | 197 |
| normal | 0.8406 | 0.9274 | 0.9098 | 0.8738 | 0.8698 | 0.9670 | 0.9573 | 276 |
| sever | 0.9917 | 0.8774 | 0.6723 | 0.8013 | 0.9976 | 0.9942 | 0.9815 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=149 FP=12 TN=384 FN=48
- **normal**: TP=232 FP=23 TN=294 FN=44
- **sever**: TP=119 FP=58 TN=415 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 149 | 22 | 26 |
| normal | 12 | 232 | 32 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.9255    0.7563    0.8324       197
      normal     0.9098    0.8406    0.8738       276
       sever     0.6723    0.9917    0.8013       120

    accuracy                         0.8432       593
   macro avg     0.8359    0.8629    0.8359       593
weighted avg     0.8669    0.8432    0.8454       593

```