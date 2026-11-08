# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz8_dce4m_seed42/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8270 |
| Accuracy | 0.8475 |
| Macro-AUROC (OvR) | 0.9614 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8471 |
| Balanced Acc | 0.8251 |
| Macro-Sensitivity | 0.8251 |
| Macro-Specificity | 0.9119 |
| Macro-Precision | 0.8290 |
| AUC-PR (Macro) | 0.9218 |
| Cohen's Kappa | 0.7381 |
| MCC | 0.7382 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1515.0 (0.66 ms/张) |
| End-to-End FPS | 12.6 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8939 | 0.8462 | 0.8806 | 0.8872 | 0.8627 | 0.9528 | 0.9665 | 66 |
| malignant | 0.7812 | 0.9302 | 0.8065 | 0.7937 | 0.9195 | 0.9600 | 0.8947 | 32 |
| normal | 0.8000 | 0.9592 | 0.8000 | 0.8000 | 0.9592 | 0.9714 | 0.9044 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=8 TN=44 FN=7
- **malignant**: TP=25 FP=6 TN=80 FN=7
- **normal**: TP=16 FP=4 TN=94 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 6 | 1 |
| malignant | 4 | 25 | 3 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8806    0.8939    0.8872        66
   malignant     0.8065    0.7812    0.7937        32
      normal     0.8000    0.8000    0.8000        20

    accuracy                         0.8475       118
   macro avg     0.8290    0.8251    0.8270       118
weighted avg     0.8468    0.8475    0.8471       118

```