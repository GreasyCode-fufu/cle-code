# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz8_dce4m_seed43/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8025 |
| Accuracy | 0.8276 |
| Macro-AUROC (OvR) | 0.9429 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8279 |
| Balanced Acc | 0.8063 |
| Macro-Sensitivity | 0.8063 |
| Macro-Specificity | 0.9012 |
| Macro-Precision | 0.8037 |
| AUC-PR (Macro) | 0.9016 |
| Cohen's Kappa | 0.7052 |
| MCC | 0.7063 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1268.8 (0.79 ms/张) |
| End-to-End FPS | 12.6 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8769 | 0.8235 | 0.8636 | 0.8702 | 0.8400 | 0.9373 | 0.9403 | 65 |
| malignant | 0.7419 | 0.9529 | 0.8519 | 0.7931 | 0.9101 | 0.9389 | 0.9045 | 31 |
| normal | 0.8000 | 0.9271 | 0.6957 | 0.7442 | 0.9570 | 0.9526 | 0.8600 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=9 TN=42 FN=8
- **malignant**: TP=23 FP=4 TN=81 FN=8
- **normal**: TP=16 FP=7 TN=89 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 4 | 4 |
| malignant | 5 | 23 | 3 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8636    0.8769    0.8702        65
   malignant     0.8519    0.7419    0.7931        31
      normal     0.6957    0.8000    0.7442        20

    accuracy                         0.8276       116
   macro avg     0.8037    0.8063    0.8025       116
weighted avg     0.8315    0.8276    0.8279       116

```