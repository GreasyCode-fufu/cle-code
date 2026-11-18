# iformer_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/iformer_s_seed42/iformer_s/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.6979 |
| Accuracy | 0.7241 |
| Macro-AUROC (OvR) | 0.8621 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7294 |
| Balanced Acc | 0.7279 |
| Macro-Sensitivity | 0.7279 |
| Macro-Specificity | 0.8573 |
| Macro-Precision | 0.6884 |
| AUC-PR (Macro) | 0.7556 |
| Cohen's Kappa | 0.5502 |
| MCC | 0.5562 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 6.24M |
| Pure-Inference FPS | 2690.7 (0.37 ms/张) |
| End-to-End FPS | 174.4 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.7385 | 0.8235 | 0.8421 | 0.7869 | 0.7119 | 0.8323 | 0.8483 | 65 |
| malignant | 0.6452 | 0.8941 | 0.6897 | 0.6667 | 0.8736 | 0.8509 | 0.6341 | 31 |
| normal | 0.8000 | 0.8542 | 0.5333 | 0.6400 | 0.9535 | 0.9031 | 0.7846 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=48 FP=9 TN=42 FN=17
- **malignant**: TP=20 FP=9 TN=76 FN=11
- **normal**: TP=16 FP=14 TN=82 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 48 | 8 | 9 |
| malignant | 6 | 20 | 5 |
| normal | 3 | 1 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8421    0.7385    0.7869        65
   malignant     0.6897    0.6452    0.6667        31
      normal     0.5333    0.8000    0.6400        20

    accuracy                         0.7241       116
   macro avg     0.6884    0.7279    0.6979       116
weighted avg     0.7481    0.7241    0.7294       116

```