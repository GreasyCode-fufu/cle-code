# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/convnext_small_seed43/convnext_small/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8539 |
| Accuracy | 0.8707 |
| Macro-AUROC (OvR) | 0.9423 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8722 |
| Balanced Acc | 0.8719 |
| Macro-Sensitivity | 0.8719 |
| Macro-Specificity | 0.9334 |
| Macro-Precision | 0.8410 |
| AUC-PR (Macro) | 0.9041 |
| Cohen's Kappa | 0.7839 |
| MCC | 0.7856 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.41M |
| Pure-Inference FPS | 907.7 (1.10 ms/张) |
| End-to-End FPS | 161.3 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8769 | 0.9216 | 0.9344 | 0.9048 | 0.8545 | 0.9385 | 0.9342 | 65 |
| malignant | 0.8387 | 0.9412 | 0.8387 | 0.8387 | 0.9412 | 0.9442 | 0.9013 | 31 |
| normal | 0.9000 | 0.9375 | 0.7500 | 0.8182 | 0.9783 | 0.9443 | 0.8768 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=4 TN=47 FN=8
- **malignant**: TP=26 FP=5 TN=80 FN=5
- **normal**: TP=18 FP=6 TN=90 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 5 | 3 |
| malignant | 2 | 26 | 3 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9344    0.8769    0.9048        65
   malignant     0.8387    0.8387    0.8387        31
      normal     0.7500    0.9000    0.8182        20

    accuracy                         0.8707       116
   macro avg     0.8410    0.8719    0.8539       116
weighted avg     0.8770    0.8707    0.8722       116

```