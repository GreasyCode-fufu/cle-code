# convnext_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_dsd_seed42/convnext_dsd/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8624 |
| Accuracy | 0.8966 |
| Macro-AUROC (OvR) | 0.9563 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8962 |
| Balanced Acc | 0.8645 |
| Macro-Sensitivity | 0.8645 |
| Macro-Specificity | 0.9452 |
| Macro-Precision | 0.8675 |
| AUC-PR (Macro) | 0.9185 |
| Cohen's Kappa | 0.8223 |
| MCC | 0.8244 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 27.78M |
| Pure-Inference FPS | 432.6 (2.31 ms/张) |
| End-to-End FPS | 130.9 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9692 | 0.9216 | 0.9403 | 0.9545 | 0.9592 | 0.9786 | 0.9716 | 65 |
| malignant | 0.7742 | 0.9765 | 0.9231 | 0.8421 | 0.9222 | 0.9423 | 0.9157 | 31 |
| normal | 0.8500 | 0.9375 | 0.7391 | 0.7907 | 0.9677 | 0.9479 | 0.8683 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=63 FP=4 TN=47 FN=2
- **malignant**: TP=24 FP=2 TN=83 FN=7
- **normal**: TP=17 FP=6 TN=90 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 63 | 1 | 1 |
| malignant | 2 | 24 | 5 |
| normal | 2 | 1 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9403    0.9692    0.9545        65
   malignant     0.9231    0.7742    0.8421        31
      normal     0.7391    0.8500    0.7907        20

    accuracy                         0.8966       116
   macro avg     0.8675    0.8645    0.8624       116
weighted avg     0.9010    0.8966    0.8962       116

```