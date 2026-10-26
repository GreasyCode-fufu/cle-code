# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/resnet50_seed44/resnet50/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8689 |
| Accuracy | 0.8668 |
| Macro-AUROC (OvR) | 0.9717 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8667 |
| Balanced Acc | 0.8811 |
| Macro-Sensitivity | 0.8811 |
| Macro-Specificity | 0.9317 |
| Macro-Precision | 0.8599 |
| AUC-PR (Macro) | 0.9580 |
| Cohen's Kappa | 0.7919 |
| MCC | 0.7933 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 1002.4 (1.00 ms/张) |
| End-to-End FPS | 277.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8528 | 0.9141 | 0.8317 | 0.8421 | 0.9258 | 0.9583 | 0.9424 | 197 |
| normal | 0.8406 | 0.9274 | 0.9098 | 0.8738 | 0.8698 | 0.9656 | 0.9562 | 276 |
| sever | 0.9500 | 0.9535 | 0.8382 | 0.8906 | 0.9869 | 0.9912 | 0.9752 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=168 FP=34 TN=362 FN=29
- **normal**: TP=232 FP=23 TN=294 FN=44
- **sever**: TP=114 FP=22 TN=451 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 168 | 17 | 12 |
| normal | 34 | 232 | 10 |
| sever | 0 | 6 | 114 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8317    0.8528    0.8421       197
      normal     0.9098    0.8406    0.8738       276
       sever     0.8382    0.9500    0.8906       120

    accuracy                         0.8668       593
   macro avg     0.8599    0.8811    0.8689       593
weighted avg     0.8694    0.8668    0.8667       593

```