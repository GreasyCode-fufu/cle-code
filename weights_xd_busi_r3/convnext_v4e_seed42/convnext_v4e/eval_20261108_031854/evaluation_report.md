# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r3/convnext_v4e_seed42/convnext_v4e/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8414 |
| Accuracy | 0.8707 |
| Macro-AUROC (OvR) | 0.9567 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8696 |
| Balanced Acc | 0.8494 |
| Macro-Sensitivity | 0.8494 |
| Macro-Specificity | 0.9286 |
| Macro-Precision | 0.8490 |
| AUC-PR (Macro) | 0.9274 |
| Cohen's Kappa | 0.7785 |
| MCC | 0.7824 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 475.9 (2.10 ms/张) |
| End-to-End FPS | 134.0 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.8824 | 0.9104 | 0.9242 | 0.9184 | 0.9680 | 0.9610 | 65 |
| malignant | 0.7097 | 0.9765 | 0.9167 | 0.8000 | 0.9022 | 0.9495 | 0.9208 | 31 |
| normal | 0.9000 | 0.9271 | 0.7200 | 0.8000 | 0.9780 | 0.9526 | 0.9004 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=6 TN=45 FN=4
- **malignant**: TP=22 FP=2 TN=83 FN=9
- **normal**: TP=18 FP=7 TN=89 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 2 | 2 |
| malignant | 4 | 22 | 5 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9104    0.9385    0.9242        65
   malignant     0.9167    0.7097    0.8000        31
      normal     0.7200    0.9000    0.8000        20

    accuracy                         0.8707       116
   macro avg     0.8490    0.8494    0.8414       116
weighted avg     0.8793    0.8707    0.8696       116

```