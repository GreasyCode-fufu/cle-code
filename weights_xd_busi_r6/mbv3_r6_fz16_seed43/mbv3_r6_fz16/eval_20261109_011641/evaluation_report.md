# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz16_seed43/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7913 |
| Accuracy | 0.8190 |
| Macro-AUROC (OvR) | 0.9335 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8245 |
| Balanced Acc | 0.8127 |
| Macro-Sensitivity | 0.8127 |
| Macro-Specificity | 0.9074 |
| Macro-Precision | 0.7908 |
| AUC-PR (Macro) | 0.8734 |
| Cohen's Kappa | 0.6994 |
| MCC | 0.7049 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1497.3 (0.67 ms/张) |
| End-to-End FPS | 162.5 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8462 | 0.8824 | 0.9016 | 0.8730 | 0.8182 | 0.9511 | 0.9577 | 65 |
| malignant | 0.7419 | 0.9647 | 0.8846 | 0.8070 | 0.9111 | 0.9150 | 0.8612 | 31 |
| normal | 0.8500 | 0.8750 | 0.5862 | 0.6939 | 0.9655 | 0.9344 | 0.8012 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=55 FP=6 TN=45 FN=10
- **malignant**: TP=23 FP=3 TN=82 FN=8
- **normal**: TP=17 FP=12 TN=84 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 55 | 3 | 7 |
| malignant | 3 | 23 | 5 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9016    0.8462    0.8730        65
   malignant     0.8846    0.7419    0.8070        31
      normal     0.5862    0.8500    0.6939        20

    accuracy                         0.8190       116
   macro avg     0.7908    0.8127    0.7913       116
weighted avg     0.8427    0.8190    0.8245       116

```