# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mobilenet_v3_large_seed42/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8457 |
| Accuracy | 0.8707 |
| Macro-AUROC (OvR) | 0.9408 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8704 |
| Balanced Acc | 0.8432 |
| Macro-Sensitivity | 0.8432 |
| Macro-Specificity | 0.9251 |
| Macro-Precision | 0.8501 |
| AUC-PR (Macro) | 0.8995 |
| Cohen's Kappa | 0.7773 |
| MCC | 0.7780 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 1543.5 (0.65 ms/张) |
| End-to-End FPS | 169.0 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9231 | 0.8627 | 0.8955 | 0.9091 | 0.8980 | 0.9448 | 0.9471 | 65 |
| malignant | 0.8065 | 0.9647 | 0.8929 | 0.8475 | 0.9318 | 0.9283 | 0.8828 | 31 |
| normal | 0.8000 | 0.9479 | 0.7619 | 0.7805 | 0.9579 | 0.9495 | 0.8685 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=7 TN=44 FN=5
- **malignant**: TP=25 FP=3 TN=82 FN=6
- **normal**: TP=16 FP=5 TN=91 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 2 | 3 |
| malignant | 4 | 25 | 2 |
| normal | 3 | 1 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8955    0.9231    0.9091        65
   malignant     0.8929    0.8065    0.8475        31
      normal     0.7619    0.8000    0.7805        20

    accuracy                         0.8707       116
   macro avg     0.8501    0.8432    0.8457       116
weighted avg     0.8718    0.8707    0.8704       116

```