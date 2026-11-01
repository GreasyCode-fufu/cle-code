# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_fgc_seed44/convnext_fgc/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8630 |
| Accuracy | 0.8879 |
| Macro-AUROC (OvR) | 0.9478 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8857 |
| Balanced Acc | 0.8478 |
| Macro-Sensitivity | 0.8478 |
| Macro-Specificity | 0.9295 |
| Macro-Precision | 0.8842 |
| AUC-PR (Macro) | 0.9147 |
| Cohen's Kappa | 0.8030 |
| MCC | 0.8067 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 490.0 (2.04 ms/张) |
| End-to-End FPS | 135.8 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9692 | 0.8431 | 0.8873 | 0.9265 | 0.9556 | 0.9606 | 0.9405 | 65 |
| malignant | 0.7742 | 0.9765 | 0.9231 | 0.8421 | 0.9222 | 0.9488 | 0.9184 | 31 |
| normal | 0.8000 | 0.9688 | 0.8421 | 0.8205 | 0.9588 | 0.9339 | 0.8853 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=63 FP=8 TN=43 FN=2
- **malignant**: TP=24 FP=2 TN=83 FN=7
- **normal**: TP=16 FP=3 TN=93 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 63 | 2 | 0 |
| malignant | 4 | 24 | 3 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8873    0.9692    0.9265        65
   malignant     0.9231    0.7742    0.8421        31
      normal     0.8421    0.8000    0.8205        20

    accuracy                         0.8879       116
   macro avg     0.8842    0.8478    0.8630       116
weighted avg     0.8891    0.8879    0.8857       116

```