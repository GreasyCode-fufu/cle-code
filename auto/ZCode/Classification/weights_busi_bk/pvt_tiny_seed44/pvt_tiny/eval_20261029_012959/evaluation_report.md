# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/pvt_tiny_seed44/pvt_tiny/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8368 |
| Accuracy | 0.8707 |
| Macro-AUROC (OvR) | 0.9490 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8712 |
| Balanced Acc | 0.8376 |
| Macro-Sensitivity | 0.8376 |
| Macro-Specificity | 0.9286 |
| Macro-Precision | 0.8431 |
| AUC-PR (Macro) | 0.8991 |
| Cohen's Kappa | 0.7779 |
| MCC | 0.7798 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2045.8 (0.49 ms/张) |
| End-to-End FPS | 178.5 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.8824 | 0.9104 | 0.9242 | 0.9184 | 0.9638 | 0.9567 | 65 |
| malignant | 0.7742 | 0.9765 | 0.9231 | 0.8421 | 0.9222 | 0.9514 | 0.9183 | 31 |
| normal | 0.8000 | 0.9271 | 0.6957 | 0.7442 | 0.9570 | 0.9318 | 0.8223 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=6 TN=45 FN=4
- **malignant**: TP=24 FP=2 TN=83 FN=7
- **normal**: TP=16 FP=7 TN=89 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 2 | 2 |
| malignant | 2 | 24 | 5 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9104    0.9385    0.9242        65
   malignant     0.9231    0.7742    0.8421        31
      normal     0.6957    0.8000    0.7442        20

    accuracy                         0.8707       116
   macro avg     0.8431    0.8376    0.8368       116
weighted avg     0.8768    0.8707    0.8712       116

```