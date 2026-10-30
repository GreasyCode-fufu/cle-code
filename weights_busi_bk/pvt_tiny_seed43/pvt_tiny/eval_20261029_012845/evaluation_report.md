# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/pvt_tiny_seed43/pvt_tiny/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8344 |
| Accuracy | 0.8707 |
| Macro-AUROC (OvR) | 0.9444 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8713 |
| Balanced Acc | 0.8376 |
| Macro-Sensitivity | 0.8376 |
| Macro-Specificity | 0.9313 |
| Macro-Precision | 0.8363 |
| AUC-PR (Macro) | 0.8929 |
| Cohen's Kappa | 0.7789 |
| MCC | 0.7802 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2051.5 (0.49 ms/张) |
| End-to-End FPS | 173.6 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.9020 | 0.9242 | 0.9313 | 0.9200 | 0.9707 | 0.9635 | 65 |
| malignant | 0.7742 | 0.9647 | 0.8889 | 0.8276 | 0.9213 | 0.9214 | 0.8978 | 31 |
| normal | 0.8000 | 0.9271 | 0.6957 | 0.7442 | 0.9570 | 0.9411 | 0.8173 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=5 TN=46 FN=4
- **malignant**: TP=24 FP=3 TN=82 FN=7
- **normal**: TP=16 FP=7 TN=89 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 3 | 1 |
| malignant | 1 | 24 | 6 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9242    0.9385    0.9313        65
   malignant     0.8889    0.7742    0.8276        31
      normal     0.6957    0.8000    0.7442        20

    accuracy                         0.8707       116
   macro avg     0.8363    0.8376    0.8344       116
weighted avg     0.8754    0.8707    0.8713       116

```