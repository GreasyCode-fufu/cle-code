# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r3/convnext_v4e_seed43/convnext_v4e/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8372 |
| Accuracy | 0.8621 |
| Macro-AUROC (OvR) | 0.9457 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8609 |
| Balanced Acc | 0.8383 |
| Macro-Sensitivity | 0.8383 |
| Macro-Specificity | 0.9212 |
| Macro-Precision | 0.8400 |
| AUC-PR (Macro) | 0.8886 |
| Cohen's Kappa | 0.7628 |
| MCC | 0.7640 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 540.1 (1.85 ms/张) |
| End-to-End FPS | 138.1 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9231 | 0.8627 | 0.8955 | 0.9091 | 0.8980 | 0.9532 | 0.9562 | 65 |
| malignant | 0.7419 | 0.9529 | 0.8519 | 0.7931 | 0.9101 | 0.9427 | 0.8933 | 31 |
| normal | 0.8500 | 0.9479 | 0.7727 | 0.8095 | 0.9681 | 0.9411 | 0.8164 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=7 TN=44 FN=5
- **malignant**: TP=23 FP=4 TN=81 FN=8
- **normal**: TP=17 FP=5 TN=91 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 4 | 1 |
| malignant | 4 | 23 | 4 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8955    0.9231    0.9091        65
   malignant     0.8519    0.7419    0.7931        31
      normal     0.7727    0.8500    0.8095        20

    accuracy                         0.8621       116
   macro avg     0.8400    0.8383    0.8372       116
weighted avg     0.8627    0.8621    0.8609       116

```