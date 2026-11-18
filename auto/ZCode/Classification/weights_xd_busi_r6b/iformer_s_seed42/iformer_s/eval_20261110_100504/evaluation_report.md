# iformer_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/iformer_s_seed42/iformer_s/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.6569 |
| Accuracy | 0.6695 |
| Macro-AUROC (OvR) | 0.8550 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.6766 |
| Balanced Acc | 0.7270 |
| Macro-Sensitivity | 0.7270 |
| Macro-Specificity | 0.8457 |
| Macro-Precision | 0.6548 |
| AUC-PR (Macro) | 0.7098 |
| Cohen's Kappa | 0.4935 |
| MCC | 0.5174 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 6.24M |
| Pure-Inference FPS | 2936.0 (0.34 ms/张) |
| End-to-End FPS | 182.5 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.6061 | 0.9038 | 0.8889 | 0.7207 | 0.6438 | 0.8303 | 0.8952 | 66 |
| malignant | 0.6250 | 0.8372 | 0.5882 | 0.6061 | 0.8571 | 0.8412 | 0.5627 | 32 |
| normal | 0.9500 | 0.7959 | 0.4872 | 0.6441 | 0.9873 | 0.8936 | 0.6713 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=40 FP=5 TN=47 FN=26
- **malignant**: TP=20 FP=14 TN=72 FN=12
- **normal**: TP=19 FP=20 TN=78 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 40 | 13 | 13 |
| malignant | 5 | 20 | 7 |
| normal | 0 | 1 | 19 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8889    0.6061    0.7207        66
   malignant     0.5882    0.6250    0.6061        32
      normal     0.4872    0.9500    0.6441        20

    accuracy                         0.6695       118
   macro avg     0.6548    0.7270    0.6569       118
weighted avg     0.7393    0.6695    0.6766       118

```