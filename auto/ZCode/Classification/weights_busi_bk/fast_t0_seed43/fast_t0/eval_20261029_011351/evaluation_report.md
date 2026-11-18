# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/fast_t0_seed43/fast_t0/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8493 |
| Accuracy | 0.8793 |
| Macro-AUROC (OvR) | 0.9371 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8781 |
| Balanced Acc | 0.8427 |
| Macro-Sensitivity | 0.8427 |
| Macro-Specificity | 0.9290 |
| Macro-Precision | 0.8612 |
| AUC-PR (Macro) | 0.8841 |
| Cohen's Kappa | 0.7903 |
| MCC | 0.7926 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 2776.6 (0.36 ms/张) |
| End-to-End FPS | 182.7 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9538 | 0.8627 | 0.8986 | 0.9254 | 0.9362 | 0.9309 | 0.9385 | 65 |
| malignant | 0.7742 | 0.9765 | 0.9231 | 0.8421 | 0.9222 | 0.9351 | 0.8650 | 31 |
| normal | 0.8000 | 0.9479 | 0.7619 | 0.7805 | 0.9579 | 0.9453 | 0.8489 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=62 FP=7 TN=44 FN=3
- **malignant**: TP=24 FP=2 TN=83 FN=7
- **normal**: TP=16 FP=5 TN=91 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 62 | 2 | 1 |
| malignant | 3 | 24 | 4 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8986    0.9538    0.9254        65
   malignant     0.9231    0.7742    0.8421        31
      normal     0.7619    0.8000    0.7805        20

    accuracy                         0.8793       116
   macro avg     0.8612    0.8427    0.8493       116
weighted avg     0.8815    0.8793    0.8781       116

```