# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_dce4m_seed42/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8369 |
| Accuracy | 0.8621 |
| Macro-AUROC (OvR) | 0.9490 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8600 |
| Balanced Acc | 0.8268 |
| Macro-Sensitivity | 0.8268 |
| Macro-Specificity | 0.9155 |
| Macro-Precision | 0.8520 |
| AUC-PR (Macro) | 0.9015 |
| Cohen's Kappa | 0.7590 |
| MCC | 0.7617 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1661.7 (0.60 ms/张) |
| End-to-End FPS | 154.2 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.8235 | 0.8714 | 0.9037 | 0.9130 | 0.9517 | 0.9569 | 65 |
| malignant | 0.7419 | 0.9647 | 0.8846 | 0.8070 | 0.9111 | 0.9541 | 0.9032 | 31 |
| normal | 0.8000 | 0.9583 | 0.8000 | 0.8000 | 0.9583 | 0.9411 | 0.8443 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=9 TN=42 FN=4
- **malignant**: TP=23 FP=3 TN=82 FN=8
- **normal**: TP=16 FP=4 TN=92 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 2 | 2 |
| malignant | 6 | 23 | 2 |
| normal | 3 | 1 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8714    0.9385    0.9037        65
   malignant     0.8846    0.7419    0.8070        31
      normal     0.8000    0.8000    0.8000        20

    accuracy                         0.8621       116
   macro avg     0.8520    0.8268    0.8369       116
weighted avg     0.8626    0.8621    0.8600       116

```