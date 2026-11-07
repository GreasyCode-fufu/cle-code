# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r3/convnext_v4e_seed42/convnext_v4e/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8385 |
| Accuracy | 0.8559 |
| Macro-AUROC (OvR) | 0.9518 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8583 |
| Balanced Acc | 0.8650 |
| Macro-Sensitivity | 0.8650 |
| Macro-Specificity | 0.9257 |
| Macro-Precision | 0.8303 |
| AUC-PR (Macro) | 0.9098 |
| Cohen's Kappa | 0.7605 |
| MCC | 0.7651 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 499.8 (2.00 ms/张) |
| End-to-End FPS | 140.2 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8636 | 0.9038 | 0.9194 | 0.8906 | 0.8393 | 0.9359 | 0.9546 | 66 |
| malignant | 0.7812 | 0.9651 | 0.8929 | 0.8333 | 0.9222 | 0.9364 | 0.8391 | 32 |
| normal | 0.9500 | 0.9082 | 0.6786 | 0.7917 | 0.9889 | 0.9832 | 0.9358 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=5 TN=47 FN=9
- **malignant**: TP=25 FP=3 TN=83 FN=7
- **normal**: TP=19 FP=9 TN=89 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 3 | 6 |
| malignant | 4 | 25 | 3 |
| normal | 1 | 0 | 19 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9194    0.8636    0.8906        66
   malignant     0.8929    0.7812    0.8333        32
      normal     0.6786    0.9500    0.7917        20

    accuracy                         0.8559       118
   macro avg     0.8303    0.8650    0.8385       118
weighted avg     0.8714    0.8559    0.8583       118

```