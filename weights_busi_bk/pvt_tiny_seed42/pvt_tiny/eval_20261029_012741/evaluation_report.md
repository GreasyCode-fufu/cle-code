# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/pvt_tiny_seed42/pvt_tiny/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8550 |
| Accuracy | 0.8729 |
| Macro-AUROC (OvR) | 0.9415 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8697 |
| Balanced Acc | 0.8295 |
| Macro-Sensitivity | 0.8295 |
| Macro-Specificity | 0.9124 |
| Macro-Precision | 0.8964 |
| AUC-PR (Macro) | 0.8908 |
| Cohen's Kappa | 0.7725 |
| MCC | 0.7823 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2067.9 (0.48 ms/张) |
| End-to-End FPS | 186.6 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9697 | 0.7692 | 0.8421 | 0.9014 | 0.9524 | 0.9350 | 0.9484 | 66 |
| malignant | 0.7188 | 0.9884 | 0.9583 | 0.8214 | 0.9043 | 0.9513 | 0.8410 | 32 |
| normal | 0.8000 | 0.9796 | 0.8889 | 0.8421 | 0.9600 | 0.9383 | 0.8829 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=64 FP=12 TN=40 FN=2
- **malignant**: TP=23 FP=1 TN=85 FN=9
- **normal**: TP=16 FP=2 TN=96 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 64 | 1 | 1 |
| malignant | 8 | 23 | 1 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8421    0.9697    0.9014        66
   malignant     0.9583    0.7188    0.8214        32
      normal     0.8889    0.8000    0.8421        20

    accuracy                         0.8729       118
   macro avg     0.8964    0.8295    0.8550       118
weighted avg     0.8816    0.8729    0.8697       118

```