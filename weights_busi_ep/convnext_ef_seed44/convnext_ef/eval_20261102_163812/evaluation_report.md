# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_ef_seed44/convnext_ef/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8832 |
| Accuracy | 0.9052 |
| Macro-AUROC (OvR) | 0.9404 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9059 |
| Balanced Acc | 0.8924 |
| Macro-Sensitivity | 0.8924 |
| Macro-Specificity | 0.9513 |
| Macro-Precision | 0.8774 |
| AUC-PR (Macro) | 0.9025 |
| Cohen's Kappa | 0.8392 |
| MCC | 0.8400 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 330.4 (3.03 ms/张) |
| End-to-End FPS | 118.0 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.9412 | 0.9531 | 0.9457 | 0.9231 | 0.9489 | 0.9234 | 65 |
| malignant | 0.8387 | 0.9647 | 0.8966 | 0.8667 | 0.9425 | 0.9317 | 0.9025 | 31 |
| normal | 0.9000 | 0.9479 | 0.7826 | 0.8372 | 0.9785 | 0.9406 | 0.8815 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=3 TN=48 FN=4
- **malignant**: TP=26 FP=3 TN=82 FN=5
- **normal**: TP=18 FP=5 TN=91 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 3 | 1 |
| malignant | 1 | 26 | 4 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9531    0.9385    0.9457        65
   malignant     0.8966    0.8387    0.8667        31
      normal     0.7826    0.9000    0.8372        20

    accuracy                         0.9052       116
   macro avg     0.8774    0.8924    0.8832       116
weighted avg     0.9086    0.9052    0.9059       116

```