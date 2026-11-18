# lsnet_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/lsnet_s_seed42/lsnet_s/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.6429 |
| Accuracy | 0.6897 |
| Macro-AUROC (OvR) | 0.8669 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.6976 |
| Balanced Acc | 0.6615 |
| Macro-Sensitivity | 0.6615 |
| Macro-Specificity | 0.8420 |
| Macro-Precision | 0.6364 |
| AUC-PR (Macro) | 0.7245 |
| Cohen's Kappa | 0.4905 |
| MCC | 0.4942 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 545.67M |
| FLOPs | 1.09G |
| Params | 15.70M |
| Pure-Inference FPS | 1450.7 (0.69 ms/张) |
| End-to-End FPS | 166.6 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.7538 | 0.8235 | 0.8448 | 0.7967 | 0.7241 | 0.8670 | 0.8961 | 65 |
| malignant | 0.5806 | 0.8588 | 0.6000 | 0.5902 | 0.8488 | 0.8406 | 0.7196 | 31 |
| normal | 0.6500 | 0.8438 | 0.4643 | 0.5417 | 0.9205 | 0.8932 | 0.5579 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=49 FP=9 TN=42 FN=16
- **malignant**: TP=18 FP=12 TN=73 FN=13
- **normal**: TP=13 FP=15 TN=81 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 49 | 9 | 7 |
| malignant | 5 | 18 | 8 |
| normal | 4 | 3 | 13 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8448    0.7538    0.7967        65
   malignant     0.6000    0.5806    0.5902        31
      normal     0.4643    0.6500    0.5417        20

    accuracy                         0.6897       116
   macro avg     0.6364    0.6615    0.6429       116
weighted avg     0.7138    0.6897    0.6976       116

```