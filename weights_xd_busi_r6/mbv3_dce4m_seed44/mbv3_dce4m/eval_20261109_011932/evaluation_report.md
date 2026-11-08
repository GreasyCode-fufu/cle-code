# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_dce4m_seed44/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8465 |
| Accuracy | 0.8879 |
| Macro-AUROC (OvR) | 0.9643 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8862 |
| Balanced Acc | 0.8360 |
| Macro-Sensitivity | 0.8360 |
| Macro-Specificity | 0.9356 |
| Macro-Precision | 0.8595 |
| AUC-PR (Macro) | 0.9301 |
| Cohen's Kappa | 0.8047 |
| MCC | 0.8063 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1688.3 (0.59 ms/张) |
| End-to-End FPS | 154.7 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9692 | 0.8824 | 0.9130 | 0.9403 | 0.9574 | 0.9677 | 0.9712 | 65 |
| malignant | 0.8387 | 0.9765 | 0.9286 | 0.8814 | 0.9432 | 0.9689 | 0.9391 | 31 |
| normal | 0.7000 | 0.9479 | 0.7368 | 0.7179 | 0.9381 | 0.9562 | 0.8801 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=63 FP=6 TN=45 FN=2
- **malignant**: TP=26 FP=2 TN=83 FN=5
- **normal**: TP=14 FP=5 TN=91 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 63 | 1 | 1 |
| malignant | 1 | 26 | 4 |
| normal | 5 | 1 | 14 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9130    0.9692    0.9403        65
   malignant     0.9286    0.8387    0.8814        31
      normal     0.7368    0.7000    0.7179        20

    accuracy                         0.8879       116
   macro avg     0.8595    0.8360    0.8465       116
weighted avg     0.8868    0.8879    0.8862       116

```