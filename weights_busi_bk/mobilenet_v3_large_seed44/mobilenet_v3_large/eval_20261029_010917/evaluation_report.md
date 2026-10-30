# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/mobilenet_v3_large_seed44/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8232 |
| Accuracy | 0.8621 |
| Macro-AUROC (OvR) | 0.9510 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8585 |
| Balanced Acc | 0.8096 |
| Macro-Sensitivity | 0.8096 |
| Macro-Specificity | 0.9160 |
| Macro-Precision | 0.8472 |
| AUC-PR (Macro) | 0.9189 |
| Cohen's Kappa | 0.7568 |
| MCC | 0.7622 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 1713.4 (0.58 ms/张) |
| End-to-End FPS | 175.7 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9692 | 0.8235 | 0.8750 | 0.9197 | 0.9545 | 0.9560 | 0.9554 | 65 |
| malignant | 0.7097 | 0.9765 | 0.9167 | 0.8000 | 0.9022 | 0.9461 | 0.9177 | 31 |
| normal | 0.7500 | 0.9479 | 0.7500 | 0.7500 | 0.9479 | 0.9510 | 0.8836 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=63 FP=9 TN=42 FN=2
- **malignant**: TP=22 FP=2 TN=83 FN=9
- **normal**: TP=15 FP=5 TN=91 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 63 | 1 | 1 |
| malignant | 5 | 22 | 4 |
| normal | 4 | 1 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8750    0.9692    0.9197        65
   malignant     0.9167    0.7097    0.8000        31
      normal     0.7500    0.7500    0.7500        20

    accuracy                         0.8621       116
   macro avg     0.8472    0.8096    0.8232       116
weighted avg     0.8646    0.8621    0.8585       116

```