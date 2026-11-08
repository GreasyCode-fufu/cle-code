# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16bn_dce4m_seed43/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8288 |
| Accuracy | 0.8534 |
| Macro-AUROC (OvR) | 0.9444 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8560 |
| Balanced Acc | 0.8445 |
| Macro-Sensitivity | 0.8445 |
| Macro-Specificity | 0.9239 |
| Macro-Precision | 0.8205 |
| AUC-PR (Macro) | 0.8917 |
| Cohen's Kappa | 0.7543 |
| MCC | 0.7562 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1427.1 (0.70 ms/张) |
| End-to-End FPS | 12.4 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8769 | 0.9020 | 0.9194 | 0.8976 | 0.8519 | 0.9448 | 0.9488 | 65 |
| malignant | 0.8065 | 0.9529 | 0.8621 | 0.8333 | 0.9310 | 0.9514 | 0.9156 | 31 |
| normal | 0.8500 | 0.9167 | 0.6800 | 0.7556 | 0.9670 | 0.9370 | 0.8108 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=5 TN=46 FN=8
- **malignant**: TP=25 FP=4 TN=81 FN=6
- **normal**: TP=17 FP=8 TN=88 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 4 | 4 |
| malignant | 2 | 25 | 4 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9194    0.8769    0.8976        65
   malignant     0.8621    0.8065    0.8333        31
      normal     0.6800    0.8500    0.7556        20

    accuracy                         0.8534       116
   macro avg     0.8205    0.8445    0.8288       116
weighted avg     0.8628    0.8534    0.8560       116

```