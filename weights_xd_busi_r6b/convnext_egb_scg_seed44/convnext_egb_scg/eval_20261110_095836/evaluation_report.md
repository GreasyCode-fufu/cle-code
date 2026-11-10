# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/convnext_egb_scg_seed44/convnext_egb_scg/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8506 |
| Accuracy | 0.8707 |
| Macro-AUROC (OvR) | 0.9519 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8695 |
| Balanced Acc | 0.8373 |
| Macro-Sensitivity | 0.8373 |
| Macro-Specificity | 0.9212 |
| Macro-Precision | 0.8678 |
| AUC-PR (Macro) | 0.9020 |
| Cohen's Kappa | 0.7750 |
| MCC | 0.7761 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 690.7 (1.45 ms/张) |
| End-to-End FPS | 147.2 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9231 | 0.8431 | 0.8824 | 0.9023 | 0.8958 | 0.9608 | 0.9360 | 65 |
| malignant | 0.8387 | 0.9412 | 0.8387 | 0.8387 | 0.9412 | 0.9651 | 0.9373 | 31 |
| normal | 0.7500 | 0.9792 | 0.8824 | 0.8108 | 0.9495 | 0.9297 | 0.8326 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=8 TN=43 FN=5
- **malignant**: TP=26 FP=5 TN=80 FN=5
- **normal**: TP=15 FP=2 TN=94 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 5 | 0 |
| malignant | 3 | 26 | 2 |
| normal | 5 | 0 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8824    0.9231    0.9023        65
   malignant     0.8387    0.8387    0.8387        31
      normal     0.8824    0.7500    0.8108        20

    accuracy                         0.8707       116
   macro avg     0.8678    0.8373    0.8506       116
weighted avg     0.8707    0.8707    0.8695       116

```