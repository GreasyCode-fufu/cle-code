# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz8m_seed43/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8139 |
| Accuracy | 0.8362 |
| Macro-AUROC (OvR) | 0.9411 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8389 |
| Balanced Acc | 0.8286 |
| Macro-Sensitivity | 0.8286 |
| Macro-Specificity | 0.9112 |
| Macro-Precision | 0.8105 |
| AUC-PR (Macro) | 0.9024 |
| Cohen's Kappa | 0.7246 |
| MCC | 0.7272 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1222.2 (0.82 ms/张) |
| End-to-End FPS | 21.2 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8615 | 0.8627 | 0.8889 | 0.8750 | 0.8302 | 0.9360 | 0.9421 | 65 |
| malignant | 0.7742 | 0.9647 | 0.8889 | 0.8276 | 0.9213 | 0.9294 | 0.8892 | 31 |
| normal | 0.8500 | 0.9062 | 0.6538 | 0.7391 | 0.9667 | 0.9578 | 0.8758 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=7 TN=44 FN=9
- **malignant**: TP=24 FP=3 TN=82 FN=7
- **normal**: TP=17 FP=9 TN=87 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 3 | 6 |
| malignant | 4 | 24 | 3 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8889    0.8615    0.8750        65
   malignant     0.8889    0.7742    0.8276        31
      normal     0.6538    0.8500    0.7391        20

    accuracy                         0.8362       116
   macro avg     0.8105    0.8286    0.8139       116
weighted avg     0.8484    0.8362    0.8389       116

```