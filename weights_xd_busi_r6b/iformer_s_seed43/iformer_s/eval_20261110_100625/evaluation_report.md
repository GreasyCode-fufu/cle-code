# iformer_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/iformer_s_seed43/iformer_s/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7340 |
| Accuracy | 0.7627 |
| Macro-AUROC (OvR) | 0.9009 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7653 |
| Balanced Acc | 0.7460 |
| Macro-Sensitivity | 0.7460 |
| Macro-Specificity | 0.8725 |
| Macro-Precision | 0.7248 |
| AUC-PR (Macro) | 0.7890 |
| Cohen's Kappa | 0.6037 |
| MCC | 0.6053 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 6.24M |
| Pure-Inference FPS | 2986.4 (0.33 ms/张) |
| End-to-End FPS | 184.0 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.7879 | 0.8269 | 0.8525 | 0.8189 | 0.7544 | 0.8816 | 0.8831 | 66 |
| malignant | 0.7500 | 0.8721 | 0.6857 | 0.7164 | 0.9036 | 0.8661 | 0.6994 | 32 |
| normal | 0.7000 | 0.9184 | 0.6364 | 0.6667 | 0.9375 | 0.9551 | 0.7844 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=52 FP=9 TN=43 FN=14
- **malignant**: TP=24 FP=11 TN=75 FN=8
- **normal**: TP=14 FP=8 TN=90 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 52 | 8 | 6 |
| malignant | 6 | 24 | 2 |
| normal | 3 | 3 | 14 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8525    0.7879    0.8189        66
   malignant     0.6857    0.7500    0.7164        32
      normal     0.6364    0.7000    0.6667        20

    accuracy                         0.7627       118
   macro avg     0.7248    0.7460    0.7340       118
weighted avg     0.7706    0.7627    0.7653       118

```