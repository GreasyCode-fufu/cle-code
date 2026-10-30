# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/repvit_m1_seed43/repvit_m1/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.4698 |
| Accuracy | 0.5431 |
| Macro-AUROC (OvR) | 0.5905 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.5333 |
| Balanced Acc | 0.4598 |
| Macro-Sensitivity | 0.4598 |
| Macro-Specificity | 0.7221 |
| Macro-Precision | 0.4927 |
| AUC-PR (Macro) | 0.4070 |
| Cohen's Kappa | 0.1826 |
| MCC | 0.1842 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 1997.7 (0.50 ms/张) |
| End-to-End FPS | 177.5 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.6923 | 0.4510 | 0.6164 | 0.6522 | 0.5349 | 0.5228 | 0.5650 | 65 |
| malignant | 0.3871 | 0.7882 | 0.4000 | 0.3934 | 0.7791 | 0.6087 | 0.3697 | 31 |
| normal | 0.3000 | 0.9271 | 0.4615 | 0.3636 | 0.8641 | 0.6401 | 0.2862 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=45 FP=28 TN=23 FN=20
- **malignant**: TP=12 FP=18 TN=67 FN=19
- **normal**: TP=6 FP=7 TN=89 FN=14

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 45 | 15 | 5 |
| malignant | 17 | 12 | 2 |
| normal | 11 | 3 | 6 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.6164    0.6923    0.6522        65
   malignant     0.4000    0.3871    0.3934        31
      normal     0.4615    0.3000    0.3636        20

    accuracy                         0.5431       116
   macro avg     0.4927    0.4598    0.4698       116
weighted avg     0.5319    0.5431    0.5333       116

```