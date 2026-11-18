# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz8m_seed42/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8398 |
| Accuracy | 0.8621 |
| Macro-AUROC (OvR) | 0.9284 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8636 |
| Balanced Acc | 0.8496 |
| Macro-Sensitivity | 0.8496 |
| Macro-Specificity | 0.9247 |
| Macro-Precision | 0.8358 |
| AUC-PR (Macro) | 0.8879 |
| Cohen's Kappa | 0.7665 |
| MCC | 0.7678 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 766.6 (1.30 ms/张) |
| End-to-End FPS | 28.5 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8923 | 0.8824 | 0.9062 | 0.8992 | 0.8654 | 0.9273 | 0.9230 | 65 |
| malignant | 0.8065 | 0.9647 | 0.8929 | 0.8475 | 0.9318 | 0.9252 | 0.8911 | 31 |
| normal | 0.8500 | 0.9271 | 0.7083 | 0.7727 | 0.9674 | 0.9328 | 0.8497 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=6 TN=45 FN=7
- **malignant**: TP=25 FP=3 TN=82 FN=6
- **normal**: TP=17 FP=7 TN=89 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 3 | 4 |
| malignant | 3 | 25 | 3 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9062    0.8923    0.8992        65
   malignant     0.8929    0.8065    0.8475        31
      normal     0.7083    0.8500    0.7727        20

    accuracy                         0.8621       116
   macro avg     0.8358    0.8496    0.8398       116
weighted avg     0.8685    0.8621    0.8636       116

```