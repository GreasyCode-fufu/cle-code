# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r4/convnext_v2c_seed42/convnext_v2c/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8687 |
| Accuracy | 0.8966 |
| Macro-AUROC (OvR) | 0.9505 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8959 |
| Balanced Acc | 0.8642 |
| Macro-Sensitivity | 0.8642 |
| Macro-Specificity | 0.9417 |
| Macro-Precision | 0.8740 |
| AUC-PR (Macro) | 0.9095 |
| Cohen's Kappa | 0.8216 |
| MCC | 0.8220 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 968.5 (1.03 ms/张) |
| End-to-End FPS | 158.1 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9538 | 0.9020 | 0.9254 | 0.9394 | 0.9388 | 0.9665 | 0.9545 | 65 |
| malignant | 0.8387 | 0.9647 | 0.8966 | 0.8667 | 0.9425 | 0.9537 | 0.9256 | 31 |
| normal | 0.8000 | 0.9583 | 0.8000 | 0.8000 | 0.9583 | 0.9313 | 0.8483 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=62 FP=5 TN=46 FN=3
- **malignant**: TP=26 FP=3 TN=82 FN=5
- **normal**: TP=16 FP=4 TN=92 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 62 | 3 | 0 |
| malignant | 1 | 26 | 4 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9254    0.9538    0.9394        65
   malignant     0.8966    0.8387    0.8667        31
      normal     0.8000    0.8000    0.8000        20

    accuracy                         0.8966       116
   macro avg     0.8740    0.8642    0.8687       116
weighted avg     0.8961    0.8966    0.8959       116

```