# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16_dce16_seed42/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8075 |
| Accuracy | 0.8448 |
| Macro-AUROC (OvR) | 0.9271 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8420 |
| Balanced Acc | 0.8053 |
| Macro-Sensitivity | 0.8053 |
| Macro-Specificity | 0.9112 |
| Macro-Precision | 0.8171 |
| AUC-PR (Macro) | 0.8726 |
| Cohen's Kappa | 0.7308 |
| MCC | 0.7336 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 1698.8 (0.59 ms/张) |
| End-to-End FPS | 155.6 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.8431 | 0.8841 | 0.9104 | 0.9149 | 0.9367 | 0.9330 | 65 |
| malignant | 0.6774 | 0.9529 | 0.8400 | 0.7500 | 0.8901 | 0.9135 | 0.8489 | 31 |
| normal | 0.8000 | 0.9375 | 0.7273 | 0.7619 | 0.9574 | 0.9313 | 0.8359 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=8 TN=43 FN=4
- **malignant**: TP=21 FP=4 TN=81 FN=10
- **normal**: TP=16 FP=6 TN=90 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 3 | 1 |
| malignant | 5 | 21 | 5 |
| normal | 3 | 1 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8841    0.9385    0.9104        65
   malignant     0.8400    0.6774    0.7500        31
      normal     0.7273    0.8000    0.7619        20

    accuracy                         0.8448       116
   macro avg     0.8171    0.8053    0.8075       116
weighted avg     0.8453    0.8448    0.8420       116

```