# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/convnext_egb_scg_seed43/convnext_egb_scg/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8331 |
| Accuracy | 0.8475 |
| Macro-AUROC (OvR) | 0.9405 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8498 |
| Balanced Acc | 0.8590 |
| Macro-Sensitivity | 0.8590 |
| Macro-Specificity | 0.9214 |
| Macro-Precision | 0.8176 |
| AUC-PR (Macro) | 0.9018 |
| Cohen's Kappa | 0.7477 |
| MCC | 0.7513 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 344.4 (2.90 ms/张) |
| End-to-End FPS | 125.7 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8333 | 0.9038 | 0.9167 | 0.8730 | 0.8103 | 0.9189 | 0.9435 | 66 |
| malignant | 0.8438 | 0.9419 | 0.8438 | 0.8438 | 0.9419 | 0.9364 | 0.8827 | 32 |
| normal | 0.9000 | 0.9184 | 0.6923 | 0.7826 | 0.9783 | 0.9663 | 0.8792 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=55 FP=5 TN=47 FN=11
- **malignant**: TP=27 FP=5 TN=81 FN=5
- **normal**: TP=18 FP=8 TN=90 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 55 | 5 | 6 |
| malignant | 3 | 27 | 2 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9167    0.8333    0.8730        66
   malignant     0.8438    0.8438    0.8438        32
      normal     0.6923    0.9000    0.7826        20

    accuracy                         0.8475       118
   macro avg     0.8176    0.8590    0.8331       118
weighted avg     0.8589    0.8475    0.8498       118

```