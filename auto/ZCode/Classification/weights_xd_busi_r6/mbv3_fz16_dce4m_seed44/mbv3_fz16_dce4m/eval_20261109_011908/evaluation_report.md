# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16_dce4m_seed44/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8176 |
| Accuracy | 0.8534 |
| Macro-AUROC (OvR) | 0.9403 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8498 |
| Balanced Acc | 0.8045 |
| Macro-Sensitivity | 0.8045 |
| Macro-Specificity | 0.9116 |
| Macro-Precision | 0.8363 |
| AUC-PR (Macro) | 0.8915 |
| Cohen's Kappa | 0.7424 |
| MCC | 0.7458 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1680.2 (0.60 ms/张) |
| End-to-End FPS | 155.1 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9538 | 0.8235 | 0.8732 | 0.9118 | 0.9333 | 0.9430 | 0.9477 | 65 |
| malignant | 0.7097 | 0.9529 | 0.8462 | 0.7719 | 0.9000 | 0.9378 | 0.8830 | 31 |
| normal | 0.7500 | 0.9583 | 0.7895 | 0.7692 | 0.9485 | 0.9401 | 0.8438 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=62 FP=9 TN=42 FN=3
- **malignant**: TP=22 FP=4 TN=81 FN=9
- **normal**: TP=15 FP=4 TN=92 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 62 | 3 | 0 |
| malignant | 5 | 22 | 4 |
| normal | 4 | 1 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8732    0.9538    0.9118        65
   malignant     0.8462    0.7097    0.7719        31
      normal     0.7895    0.7500    0.7692        20

    accuracy                         0.8534       116
   macro avg     0.8363    0.8045    0.8176       116
weighted avg     0.8516    0.8534    0.8498       116

```