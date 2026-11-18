# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_ef_seed44/convnext_ef/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8303 |
| Accuracy | 0.8390 |
| Macro-AUROC (OvR) | 0.9395 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8413 |
| Balanced Acc | 0.8540 |
| Macro-Sensitivity | 0.8540 |
| Macro-Specificity | 0.9186 |
| Macro-Precision | 0.8145 |
| AUC-PR (Macro) | 0.8957 |
| Cohen's Kappa | 0.7347 |
| MCC | 0.7394 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 353.1 (2.83 ms/张) |
| End-to-End FPS | 127.2 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8182 | 0.9231 | 0.9310 | 0.8710 | 0.8000 | 0.9158 | 0.9450 | 66 |
| malignant | 0.8438 | 0.8837 | 0.7297 | 0.7826 | 0.9383 | 0.9222 | 0.8239 | 32 |
| normal | 0.9000 | 0.9490 | 0.7826 | 0.8372 | 0.9789 | 0.9806 | 0.9181 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=54 FP=4 TN=48 FN=12
- **malignant**: TP=27 FP=10 TN=76 FN=5
- **normal**: TP=18 FP=5 TN=93 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 54 | 9 | 3 |
| malignant | 3 | 27 | 2 |
| normal | 1 | 1 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9310    0.8182    0.8710        66
   malignant     0.7297    0.8438    0.7826        32
      normal     0.7826    0.9000    0.8372        20

    accuracy                         0.8390       118
   macro avg     0.8145    0.8540    0.8303       118
weighted avg     0.8513    0.8390    0.8413       118

```