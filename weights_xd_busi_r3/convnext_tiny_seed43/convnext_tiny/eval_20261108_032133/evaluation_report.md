# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r3/convnext_tiny_seed43/convnext_tiny/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8837 |
| Accuracy | 0.9052 |
| Macro-AUROC (OvR) | 0.9601 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9047 |
| Balanced Acc | 0.8868 |
| Macro-Sensitivity | 0.8868 |
| Macro-Specificity | 0.9482 |
| Macro-Precision | 0.8835 |
| AUC-PR (Macro) | 0.9239 |
| Cohen's Kappa | 0.8376 |
| MCC | 0.8384 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1352.1 (0.74 ms/张) |
| End-to-End FPS | 170.8 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9538 | 0.9216 | 0.9394 | 0.9466 | 0.9400 | 0.9688 | 0.9341 | 65 |
| malignant | 0.8065 | 0.9647 | 0.8929 | 0.8475 | 0.9318 | 0.9662 | 0.9327 | 31 |
| normal | 0.9000 | 0.9583 | 0.8182 | 0.8571 | 0.9787 | 0.9453 | 0.9048 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=62 FP=4 TN=47 FN=3
- **malignant**: TP=25 FP=3 TN=82 FN=6
- **normal**: TP=18 FP=4 TN=92 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 62 | 3 | 0 |
| malignant | 2 | 25 | 4 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9394    0.9538    0.9466        65
   malignant     0.8929    0.8065    0.8475        31
      normal     0.8182    0.9000    0.8571        20

    accuracy                         0.9052       116
   macro avg     0.8835    0.8868    0.8837       116
weighted avg     0.9061    0.9052    0.9047       116

```