# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/pvt_tiny_seed42/pvt_tiny/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8253 |
| Accuracy | 0.8621 |
| Macro-AUROC (OvR) | 0.9460 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8567 |
| Balanced Acc | 0.8040 |
| Macro-Sensitivity | 0.8040 |
| Macro-Specificity | 0.9103 |
| Macro-Precision | 0.8658 |
| AUC-PR (Macro) | 0.8973 |
| Cohen's Kappa | 0.7532 |
| MCC | 0.7637 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2053.5 (0.49 ms/张) |
| End-to-End FPS | 179.0 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9846 | 0.7843 | 0.8533 | 0.9143 | 0.9756 | 0.9662 | 0.9661 | 65 |
| malignant | 0.6774 | 0.9882 | 0.9545 | 0.7925 | 0.8936 | 0.9431 | 0.9171 | 31 |
| normal | 0.7500 | 0.9583 | 0.7895 | 0.7692 | 0.9485 | 0.9286 | 0.8087 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=64 FP=11 TN=40 FN=1
- **malignant**: TP=21 FP=1 TN=84 FN=10
- **normal**: TP=15 FP=4 TN=92 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 64 | 1 | 0 |
| malignant | 6 | 21 | 4 |
| normal | 5 | 0 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8533    0.9846    0.9143        65
   malignant     0.9545    0.6774    0.7925        31
      normal     0.7895    0.7500    0.7692        20

    accuracy                         0.8621       116
   macro avg     0.8658    0.8040    0.8253       116
weighted avg     0.8694    0.8621    0.8567       116

```