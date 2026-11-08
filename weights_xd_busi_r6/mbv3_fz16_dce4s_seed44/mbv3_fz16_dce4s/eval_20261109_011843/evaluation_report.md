# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16_dce4s_seed44/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8325 |
| Accuracy | 0.8621 |
| Macro-AUROC (OvR) | 0.9372 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8588 |
| Balanced Acc | 0.8212 |
| Macro-Sensitivity | 0.8212 |
| Macro-Specificity | 0.9155 |
| Macro-Precision | 0.8511 |
| AUC-PR (Macro) | 0.8936 |
| Cohen's Kappa | 0.7579 |
| MCC | 0.7618 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1707.4 (0.59 ms/张) |
| End-to-End FPS | 149.0 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9538 | 0.8235 | 0.8732 | 0.9118 | 0.9333 | 0.9427 | 0.9461 | 65 |
| malignant | 0.7097 | 0.9647 | 0.8800 | 0.7857 | 0.9011 | 0.9313 | 0.8813 | 31 |
| normal | 0.8000 | 0.9583 | 0.8000 | 0.8000 | 0.9583 | 0.9375 | 0.8534 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=62 FP=9 TN=42 FN=3
- **malignant**: TP=22 FP=3 TN=82 FN=9
- **normal**: TP=16 FP=4 TN=92 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 62 | 3 | 0 |
| malignant | 5 | 22 | 4 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8732    0.9538    0.9118        65
   malignant     0.8800    0.7097    0.7857        31
      normal     0.8000    0.8000    0.8000        20

    accuracy                         0.8621       116
   macro avg     0.8511    0.8212    0.8325       116
weighted avg     0.8624    0.8621    0.8588       116

```