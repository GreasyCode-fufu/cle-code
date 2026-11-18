# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16_dce16_seed44/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8081 |
| Accuracy | 0.8448 |
| Macro-AUROC (OvR) | 0.9361 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8402 |
| Balanced Acc | 0.7938 |
| Macro-Sensitivity | 0.7938 |
| Macro-Specificity | 0.9051 |
| Macro-Precision | 0.8302 |
| AUC-PR (Macro) | 0.8803 |
| Cohen's Kappa | 0.7260 |
| MCC | 0.7307 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 1681.9 (0.59 ms/张) |
| End-to-End FPS | 154.5 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9538 | 0.8039 | 0.8611 | 0.9051 | 0.9318 | 0.9409 | 0.9449 | 65 |
| malignant | 0.6774 | 0.9529 | 0.8400 | 0.7500 | 0.8901 | 0.9283 | 0.8744 | 31 |
| normal | 0.7500 | 0.9583 | 0.7895 | 0.7692 | 0.9485 | 0.9391 | 0.8216 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=62 FP=10 TN=41 FN=3
- **malignant**: TP=21 FP=4 TN=81 FN=10
- **normal**: TP=15 FP=4 TN=92 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 62 | 3 | 0 |
| malignant | 6 | 21 | 4 |
| normal | 4 | 1 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8611    0.9538    0.9051        65
   malignant     0.8400    0.6774    0.7500        31
      normal     0.7895    0.7500    0.7692        20

    accuracy                         0.8448       116
   macro avg     0.8302    0.7938    0.8081       116
weighted avg     0.8431    0.8448    0.8402       116

```