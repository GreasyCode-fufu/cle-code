# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/ghostnet_seed42/ghostnet/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7970 |
| Accuracy | 0.8190 |
| Macro-AUROC (OvR) | 0.9418 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8212 |
| Balanced Acc | 0.8124 |
| Macro-Sensitivity | 0.8124 |
| Macro-Specificity | 0.9029 |
| Macro-Precision | 0.7861 |
| AUC-PR (Macro) | 0.8902 |
| Cohen's Kappa | 0.6974 |
| MCC | 0.6989 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 1327.0 (0.75 ms/张) |
| End-to-End FPS | 168.5 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8308 | 0.8627 | 0.8852 | 0.8571 | 0.8000 | 0.9348 | 0.9504 | 65 |
| malignant | 0.8065 | 0.9294 | 0.8065 | 0.8065 | 0.9294 | 0.9446 | 0.8891 | 31 |
| normal | 0.8000 | 0.9167 | 0.6667 | 0.7273 | 0.9565 | 0.9458 | 0.8309 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=54 FP=7 TN=44 FN=11
- **malignant**: TP=25 FP=6 TN=79 FN=6
- **normal**: TP=16 FP=8 TN=88 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 54 | 6 | 5 |
| malignant | 3 | 25 | 3 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8852    0.8308    0.8571        65
   malignant     0.8065    0.8065    0.8065        31
      normal     0.6667    0.8000    0.7273        20

    accuracy                         0.8190       116
   macro avg     0.7861    0.8124    0.7970       116
weighted avg     0.8265    0.8190    0.8212       116

```