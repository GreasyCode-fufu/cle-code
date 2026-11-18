# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/repvit_m1_seed42/repvit_m1/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.2818 |
| Accuracy | 0.3793 |
| Macro-AUROC (OvR) | 0.4957 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.3759 |
| Balanced Acc | 0.2878 |
| Macro-Sensitivity | 0.2878 |
| Macro-Specificity | 0.6472 |
| Macro-Precision | 0.2805 |
| AUC-PR (Macro) | 0.3579 |
| Cohen's Kappa | -0.0625 |
| MCC | -0.0630 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 2022.7 (0.49 ms/张) |
| End-to-End FPS | 178.3 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.5231 | 0.4314 | 0.5397 | 0.5312 | 0.4151 | 0.4926 | 0.5482 | 65 |
| malignant | 0.2903 | 0.6353 | 0.2250 | 0.2535 | 0.7105 | 0.4725 | 0.3132 | 31 |
| normal | 0.0500 | 0.8750 | 0.0769 | 0.0606 | 0.8155 | 0.5219 | 0.2124 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=34 FP=29 TN=22 FN=31
- **malignant**: TP=9 FP=31 TN=54 FN=22
- **normal**: TP=1 FP=12 TN=84 FN=19

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 34 | 20 | 11 |
| malignant | 21 | 9 | 1 |
| normal | 8 | 11 | 1 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.5397    0.5231    0.5312        65
   malignant     0.2250    0.2903    0.2535        31
      normal     0.0769    0.0500    0.0606        20

    accuracy                         0.3793       116
   macro avg     0.2805    0.2878    0.2818       116
weighted avg     0.3758    0.3793    0.3759       116

```