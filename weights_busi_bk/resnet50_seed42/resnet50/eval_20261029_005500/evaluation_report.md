# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/resnet50_seed42/resnet50/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8059 |
| Accuracy | 0.8220 |
| Macro-AUROC (OvR) | 0.9310 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8219 |
| Balanced Acc | 0.8037 |
| Macro-Sensitivity | 0.8037 |
| Macro-Specificity | 0.8952 |
| Macro-Precision | 0.8086 |
| AUC-PR (Macro) | 0.8693 |
| Cohen's Kappa | 0.6953 |
| MCC | 0.6954 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 1470.0 (0.68 ms/张) |
| End-to-End FPS | 180.6 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8485 | 0.8077 | 0.8485 | 0.8485 | 0.8077 | 0.8951 | 0.9245 | 66 |
| malignant | 0.8125 | 0.9186 | 0.7879 | 0.8000 | 0.9294 | 0.9346 | 0.8401 | 32 |
| normal | 0.7500 | 0.9592 | 0.7895 | 0.7692 | 0.9495 | 0.9633 | 0.8434 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=10 TN=42 FN=10
- **malignant**: TP=26 FP=7 TN=79 FN=6
- **normal**: TP=15 FP=4 TN=94 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 7 | 3 |
| malignant | 5 | 26 | 1 |
| normal | 5 | 0 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8485    0.8485    0.8485        66
   malignant     0.7879    0.8125    0.8000        32
      normal     0.7895    0.7500    0.7692        20

    accuracy                         0.8220       118
   macro avg     0.8086    0.8037    0.8059       118
weighted avg     0.8220    0.8220    0.8219       118

```