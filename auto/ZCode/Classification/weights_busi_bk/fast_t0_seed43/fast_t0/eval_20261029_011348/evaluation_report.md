# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/fast_t0_seed43/fast_t0/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8481 |
| Accuracy | 0.8644 |
| Macro-AUROC (OvR) | 0.9480 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8631 |
| Balanced Acc | 0.8352 |
| Macro-Sensitivity | 0.8352 |
| Macro-Specificity | 0.9141 |
| Macro-Precision | 0.8647 |
| AUC-PR (Macro) | 0.9052 |
| Cohen's Kappa | 0.7629 |
| MCC | 0.7651 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 2812.0 (0.36 ms/张) |
| End-to-End FPS | 190.8 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9242 | 0.8077 | 0.8592 | 0.8905 | 0.8936 | 0.9376 | 0.9519 | 66 |
| malignant | 0.7812 | 0.9651 | 0.8929 | 0.8333 | 0.9222 | 0.9350 | 0.8797 | 32 |
| normal | 0.8000 | 0.9694 | 0.8421 | 0.8205 | 0.9596 | 0.9714 | 0.8841 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=10 TN=42 FN=5
- **malignant**: TP=25 FP=3 TN=83 FN=7
- **normal**: TP=16 FP=3 TN=95 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 3 | 2 |
| malignant | 6 | 25 | 1 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8592    0.9242    0.8905        66
   malignant     0.8929    0.7812    0.8333        32
      normal     0.8421    0.8000    0.8205        20

    accuracy                         0.8644       118
   macro avg     0.8647    0.8352    0.8481       118
weighted avg     0.8654    0.8644    0.8631       118

```