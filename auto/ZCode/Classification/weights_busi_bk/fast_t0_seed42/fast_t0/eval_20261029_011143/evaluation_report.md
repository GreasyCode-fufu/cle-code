# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/fast_t0_seed42/fast_t0/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8189 |
| Accuracy | 0.8534 |
| Macro-AUROC (OvR) | 0.9287 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8521 |
| Balanced Acc | 0.8101 |
| Macro-Sensitivity | 0.8101 |
| Macro-Specificity | 0.9125 |
| Macro-Precision | 0.8352 |
| AUC-PR (Macro) | 0.8790 |
| Cohen's Kappa | 0.7443 |
| MCC | 0.7475 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 2759.0 (0.36 ms/张) |
| End-to-End FPS | 182.9 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.8235 | 0.8714 | 0.9037 | 0.9130 | 0.9270 | 0.9390 | 65 |
| malignant | 0.7419 | 0.9765 | 0.9200 | 0.8214 | 0.9121 | 0.9298 | 0.8633 | 31 |
| normal | 0.7500 | 0.9375 | 0.7143 | 0.7317 | 0.9474 | 0.9292 | 0.8346 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=9 TN=42 FN=4
- **malignant**: TP=23 FP=2 TN=83 FN=8
- **normal**: TP=15 FP=6 TN=90 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 2 | 2 |
| malignant | 4 | 23 | 4 |
| normal | 5 | 0 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8714    0.9385    0.9037        65
   malignant     0.9200    0.7419    0.8214        31
      normal     0.7143    0.7500    0.7317        20

    accuracy                         0.8534       116
   macro avg     0.8352    0.8101    0.8189       116
weighted avg     0.8573    0.8534    0.8521       116

```