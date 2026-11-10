# iformer_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/iformer_s_seed44/iformer_s/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7593 |
| Accuracy | 0.7931 |
| Macro-AUROC (OvR) | 0.9100 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7932 |
| Balanced Acc | 0.7571 |
| Macro-Sensitivity | 0.7571 |
| Macro-Specificity | 0.8846 |
| Macro-Precision | 0.7620 |
| AUC-PR (Macro) | 0.8542 |
| Cohen's Kappa | 0.6458 |
| MCC | 0.6458 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 6.24M |
| Pure-Inference FPS | 2919.2 (0.34 ms/张) |
| End-to-End FPS | 176.4 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8615 | 0.8235 | 0.8615 | 0.8615 | 0.8235 | 0.9155 | 0.9248 | 65 |
| malignant | 0.7097 | 0.8824 | 0.6875 | 0.6984 | 0.8929 | 0.8839 | 0.7991 | 31 |
| normal | 0.7000 | 0.9479 | 0.7368 | 0.7179 | 0.9381 | 0.9307 | 0.8386 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=9 TN=42 FN=9
- **malignant**: TP=22 FP=10 TN=75 FN=9
- **normal**: TP=14 FP=5 TN=91 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 7 | 2 |
| malignant | 6 | 22 | 3 |
| normal | 3 | 3 | 14 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8615    0.8615    0.8615        65
   malignant     0.6875    0.7097    0.6984        31
      normal     0.7368    0.7000    0.7179        20

    accuracy                         0.7931       116
   macro avg     0.7620    0.7571    0.7593       116
weighted avg     0.7935    0.7931    0.7932       116

```