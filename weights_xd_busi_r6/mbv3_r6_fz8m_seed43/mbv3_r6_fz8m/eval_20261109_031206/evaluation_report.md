# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz8m_seed43/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8270 |
| Accuracy | 0.8475 |
| Macro-AUROC (OvR) | 0.9520 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8465 |
| Balanced Acc | 0.8134 |
| Macro-Sensitivity | 0.8134 |
| Macro-Specificity | 0.9084 |
| Macro-Precision | 0.8444 |
| AUC-PR (Macro) | 0.9211 |
| Cohen's Kappa | 0.7347 |
| MCC | 0.7357 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1138.8 (0.88 ms/张) |
| End-to-End FPS | 19.3 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9091 | 0.8269 | 0.8696 | 0.8889 | 0.8776 | 0.9446 | 0.9535 | 66 |
| malignant | 0.7812 | 0.9186 | 0.7812 | 0.7812 | 0.9186 | 0.9582 | 0.8988 | 32 |
| normal | 0.7500 | 0.9796 | 0.8824 | 0.8108 | 0.9505 | 0.9531 | 0.9111 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=9 TN=43 FN=6
- **malignant**: TP=25 FP=7 TN=79 FN=7
- **normal**: TP=15 FP=2 TN=96 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 6 | 0 |
| malignant | 5 | 25 | 2 |
| normal | 4 | 1 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8696    0.9091    0.8889        66
   malignant     0.7812    0.7812    0.7812        32
      normal     0.8824    0.7500    0.8108        20

    accuracy                         0.8475       118
   macro avg     0.8444    0.8134    0.8270       118
weighted avg     0.8478    0.8475    0.8465       118

```