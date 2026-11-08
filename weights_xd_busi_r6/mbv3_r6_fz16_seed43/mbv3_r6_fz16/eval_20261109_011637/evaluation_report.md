# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz16_seed43/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7903 |
| Accuracy | 0.8136 |
| Macro-AUROC (OvR) | 0.9491 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8187 |
| Balanced Acc | 0.8227 |
| Macro-Sensitivity | 0.8227 |
| Macro-Specificity | 0.9078 |
| Macro-Precision | 0.7809 |
| AUC-PR (Macro) | 0.9100 |
| Cohen's Kappa | 0.6947 |
| MCC | 0.7016 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1489.5 (0.67 ms/张) |
| End-to-End FPS | 167.7 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8182 | 0.9038 | 0.9153 | 0.8640 | 0.7966 | 0.9356 | 0.9552 | 66 |
| malignant | 0.7500 | 0.9419 | 0.8276 | 0.7869 | 0.9101 | 0.9469 | 0.8911 | 32 |
| normal | 0.9000 | 0.8776 | 0.6000 | 0.7200 | 0.9773 | 0.9648 | 0.8836 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=54 FP=5 TN=47 FN=12
- **malignant**: TP=24 FP=5 TN=81 FN=8
- **normal**: TP=18 FP=12 TN=86 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 54 | 5 | 7 |
| malignant | 3 | 24 | 5 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9153    0.8182    0.8640        66
   malignant     0.8276    0.7500    0.7869        32
      normal     0.6000    0.9000    0.7200        20

    accuracy                         0.8136       118
   macro avg     0.7809    0.8227    0.7903       118
weighted avg     0.8380    0.8136    0.8187       118

```