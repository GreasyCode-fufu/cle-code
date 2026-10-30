# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/resnet50_seed43/resnet50/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8627 |
| Accuracy | 0.8729 |
| Macro-AUROC (OvR) | 0.9423 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8724 |
| Balanced Acc | 0.8509 |
| Macro-Sensitivity | 0.8509 |
| Macro-Specificity | 0.9200 |
| Macro-Precision | 0.8765 |
| AUC-PR (Macro) | 0.8934 |
| Cohen's Kappa | 0.7792 |
| MCC | 0.7800 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 1461.4 (0.68 ms/张) |
| End-to-End FPS | 179.9 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9091 | 0.8269 | 0.8696 | 0.8889 | 0.8776 | 0.9175 | 0.9413 | 66 |
| malignant | 0.8438 | 0.9535 | 0.8710 | 0.8571 | 0.9425 | 0.9415 | 0.8499 | 32 |
| normal | 0.8000 | 0.9796 | 0.8889 | 0.8421 | 0.9600 | 0.9679 | 0.8890 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=9 TN=43 FN=6
- **malignant**: TP=27 FP=4 TN=82 FN=5
- **normal**: TP=16 FP=2 TN=96 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 4 | 2 |
| malignant | 5 | 27 | 0 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8696    0.9091    0.8889        66
   malignant     0.8710    0.8438    0.8571        32
      normal     0.8889    0.8000    0.8421        20

    accuracy                         0.8729       118
   macro avg     0.8765    0.8509    0.8627       118
weighted avg     0.8732    0.8729    0.8724       118

```