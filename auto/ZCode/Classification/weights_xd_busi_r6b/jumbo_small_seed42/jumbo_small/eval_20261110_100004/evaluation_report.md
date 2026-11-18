# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/jumbo_small_seed42/jumbo_small/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8566 |
| Accuracy | 0.8793 |
| Macro-AUROC (OvR) | 0.9462 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8821 |
| Balanced Acc | 0.8829 |
| Macro-Sensitivity | 0.8829 |
| Macro-Specificity | 0.9435 |
| Macro-Precision | 0.8441 |
| AUC-PR (Macro) | 0.8829 |
| Cohen's Kappa | 0.7999 |
| MCC | 0.8041 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 1152.6 (0.87 ms/张) |
| End-to-End FPS | 161.4 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8923 | 0.9608 | 0.9667 | 0.9280 | 0.8750 | 0.9535 | 0.9255 | 65 |
| malignant | 0.8065 | 0.9529 | 0.8621 | 0.8333 | 0.9310 | 0.9408 | 0.9027 | 31 |
| normal | 0.9500 | 0.9167 | 0.7037 | 0.8085 | 0.9888 | 0.9443 | 0.8204 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=2 TN=49 FN=7
- **malignant**: TP=25 FP=4 TN=81 FN=6
- **normal**: TP=19 FP=8 TN=88 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 4 | 3 |
| malignant | 1 | 25 | 5 |
| normal | 1 | 0 | 19 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9667    0.8923    0.9280        65
   malignant     0.8621    0.8065    0.8333        31
      normal     0.7037    0.9500    0.8085        20

    accuracy                         0.8793       116
   macro avg     0.8441    0.8829    0.8566       116
weighted avg     0.8934    0.8793    0.8821       116

```