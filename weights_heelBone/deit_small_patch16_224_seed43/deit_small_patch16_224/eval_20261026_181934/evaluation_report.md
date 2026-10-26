# deit_small_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/deit_small_patch16_224_seed43/deit_small_patch16_224/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8251 |
| Accuracy | 0.8212 |
| Macro-AUROC (OvR) | 0.9448 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8189 |
| Balanced Acc | 0.8406 |
| Macro-Sensitivity | 0.8406 |
| Macro-Specificity | 0.9055 |
| Macro-Precision | 0.8170 |
| AUC-PR (Macro) | 0.9231 |
| Cohen's Kappa | 0.7201 |
| MCC | 0.7220 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.25G |
| FLOPs | 8.50G |
| Params | 21.59M |
| Pure-Inference FPS | 1103.2 (0.91 ms/张) |
| End-to-End FPS | 285.2 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7208 | 0.9167 | 0.8114 | 0.7634 | 0.8684 | 0.9112 | 0.8749 | 197 |
| normal | 0.8261 | 0.8612 | 0.8382 | 0.8321 | 0.8505 | 0.9304 | 0.9154 | 276 |
| sever | 0.9750 | 0.9387 | 0.8014 | 0.8797 | 0.9933 | 0.9927 | 0.9791 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=142 FP=33 TN=363 FN=55
- **normal**: TP=228 FP=44 TN=273 FN=48
- **sever**: TP=117 FP=29 TN=444 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 142 | 43 | 12 |
| normal | 31 | 228 | 17 |
| sever | 2 | 1 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8114    0.7208    0.7634       197
      normal     0.8382    0.8261    0.8321       276
       sever     0.8014    0.9750    0.8797       120

    accuracy                         0.8212       593
   macro avg     0.8170    0.8406    0.8251       593
weighted avg     0.8219    0.8212    0.8189       593

```