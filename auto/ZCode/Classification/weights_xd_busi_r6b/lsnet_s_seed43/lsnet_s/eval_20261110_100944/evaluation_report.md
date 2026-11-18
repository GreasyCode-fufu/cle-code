# lsnet_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/lsnet_s_seed43/lsnet_s/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7453 |
| Accuracy | 0.7845 |
| Macro-AUROC (OvR) | 0.8958 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7849 |
| Balanced Acc | 0.7522 |
| Macro-Sensitivity | 0.7522 |
| Macro-Specificity | 0.8803 |
| Macro-Precision | 0.7495 |
| AUC-PR (Macro) | 0.7963 |
| Cohen's Kappa | 0.6325 |
| MCC | 0.6350 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 545.67M |
| FLOPs | 1.09G |
| Params | 15.70M |
| Pure-Inference FPS | 1639.9 (0.61 ms/张) |
| End-to-End FPS | 174.3 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8615 | 0.8039 | 0.8485 | 0.8550 | 0.8200 | 0.9071 | 0.9013 | 65 |
| malignant | 0.6452 | 0.9412 | 0.8000 | 0.7143 | 0.8791 | 0.8903 | 0.8210 | 31 |
| normal | 0.7500 | 0.8958 | 0.6000 | 0.6667 | 0.9451 | 0.8901 | 0.6666 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=10 TN=41 FN=9
- **malignant**: TP=20 FP=5 TN=80 FN=11
- **normal**: TP=15 FP=10 TN=86 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 4 | 5 |
| malignant | 6 | 20 | 5 |
| normal | 4 | 1 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8485    0.8615    0.8550        65
   malignant     0.8000    0.6452    0.7143        31
      normal     0.6000    0.7500    0.6667        20

    accuracy                         0.7845       116
   macro avg     0.7495    0.7522    0.7453       116
weighted avg     0.7927    0.7845    0.7849       116

```