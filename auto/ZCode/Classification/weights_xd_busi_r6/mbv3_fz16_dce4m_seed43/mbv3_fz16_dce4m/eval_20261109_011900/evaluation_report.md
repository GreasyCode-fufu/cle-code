# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16_dce4m_seed43/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8279 |
| Accuracy | 0.8707 |
| Macro-AUROC (OvR) | 0.9405 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8674 |
| Balanced Acc | 0.8145 |
| Macro-Sensitivity | 0.8145 |
| Macro-Specificity | 0.9247 |
| Macro-Precision | 0.8450 |
| AUC-PR (Macro) | 0.8872 |
| Cohen's Kappa | 0.7734 |
| MCC | 0.7756 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1682.0 (0.59 ms/张) |
| End-to-End FPS | 157.1 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9692 | 0.8627 | 0.9000 | 0.9333 | 0.9565 | 0.9466 | 0.9535 | 65 |
| malignant | 0.7742 | 0.9529 | 0.8571 | 0.8136 | 0.9205 | 0.9469 | 0.8832 | 31 |
| normal | 0.7000 | 0.9583 | 0.7778 | 0.7368 | 0.9388 | 0.9281 | 0.8249 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=63 FP=7 TN=44 FN=2
- **malignant**: TP=24 FP=4 TN=81 FN=7
- **normal**: TP=14 FP=4 TN=92 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 63 | 2 | 0 |
| malignant | 3 | 24 | 4 |
| normal | 4 | 2 | 14 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9000    0.9692    0.9333        65
   malignant     0.8571    0.7742    0.8136        31
      normal     0.7778    0.7000    0.7368        20

    accuracy                         0.8707       116
   macro avg     0.8450    0.8145    0.8279       116
weighted avg     0.8675    0.8707    0.8674       116

```