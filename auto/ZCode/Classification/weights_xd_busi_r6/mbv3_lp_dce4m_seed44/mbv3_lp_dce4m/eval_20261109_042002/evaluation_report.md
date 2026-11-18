# mbv3_lp_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_lp_dce4m_seed44/mbv3_lp_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7770 |
| Accuracy | 0.8017 |
| Macro-AUROC (OvR) | 0.9121 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8026 |
| Balanced Acc | 0.7853 |
| Macro-Sensitivity | 0.7853 |
| Macro-Specificity | 0.8916 |
| Macro-Precision | 0.7700 |
| AUC-PR (Macro) | 0.8638 |
| Cohen's Kappa | 0.6648 |
| MCC | 0.6652 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 946.0 (1.06 ms/张) |
| End-to-End FPS | 15.0 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8462 | 0.8431 | 0.8730 | 0.8594 | 0.8113 | 0.9053 | 0.9141 | 65 |
| malignant | 0.7097 | 0.8941 | 0.7097 | 0.7097 | 0.8941 | 0.9013 | 0.8472 | 31 |
| normal | 0.8000 | 0.9375 | 0.7273 | 0.7619 | 0.9574 | 0.9297 | 0.8301 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=55 FP=8 TN=43 FN=10
- **malignant**: TP=22 FP=9 TN=76 FN=9
- **normal**: TP=16 FP=6 TN=90 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 55 | 8 | 2 |
| malignant | 5 | 22 | 4 |
| normal | 3 | 1 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8730    0.8462    0.8594        65
   malignant     0.7097    0.7097    0.7097        31
      normal     0.7273    0.8000    0.7619        20

    accuracy                         0.8017       116
   macro avg     0.7700    0.7853    0.7770       116
weighted avg     0.8042    0.8017    0.8026       116

```