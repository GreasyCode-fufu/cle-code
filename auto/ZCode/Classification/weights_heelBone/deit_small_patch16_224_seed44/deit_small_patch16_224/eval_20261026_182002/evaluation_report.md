# deit_small_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/deit_small_patch16_224_seed44/deit_small_patch16_224/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7609 |
| Accuracy | 0.7386 |
| Macro-AUROC (OvR) | 0.9408 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7293 |
| Balanced Acc | 0.7995 |
| Macro-Sensitivity | 0.7995 |
| Macro-Specificity | 0.8695 |
| Macro-Precision | 0.7786 |
| AUC-PR (Macro) | 0.9142 |
| Cohen's Kappa | 0.6057 |
| MCC | 0.6368 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.25G |
| FLOPs | 8.50G |
| Params | 21.59M |
| Pure-Inference FPS | 951.0 (1.05 ms/张) |
| End-to-End FPS | 276.7 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8934 | 0.7096 | 0.6048 | 0.7213 | 0.9305 | 0.9147 | 0.8726 | 197 |
| normal | 0.5217 | 0.9495 | 0.9000 | 0.6606 | 0.6952 | 0.9129 | 0.8888 | 276 |
| sever | 0.9833 | 0.9493 | 0.8310 | 0.9008 | 0.9956 | 0.9948 | 0.9811 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=176 FP=115 TN=281 FN=21
- **normal**: TP=144 FP=16 TN=301 FN=132
- **sever**: TP=118 FP=24 TN=449 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 176 | 14 | 7 |
| normal | 115 | 144 | 17 |
| sever | 0 | 2 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6048    0.8934    0.7213       197
      normal     0.9000    0.5217    0.6606       276
       sever     0.8310    0.9833    0.9008       120

    accuracy                         0.7386       593
   macro avg     0.7786    0.7995    0.7609       593
weighted avg     0.7880    0.7386    0.7293       593

```