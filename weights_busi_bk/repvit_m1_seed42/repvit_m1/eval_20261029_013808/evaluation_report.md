# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/repvit_m1_seed42/repvit_m1/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.4563 |
| Accuracy | 0.5593 |
| Macro-AUROC (OvR) | 0.6214 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.5470 |
| Balanced Acc | 0.4549 |
| Macro-Sensitivity | 0.4549 |
| Macro-Specificity | 0.7419 |
| Macro-Precision | 0.4662 |
| AUC-PR (Macro) | 0.4365 |
| Cohen's Kappa | 0.2189 |
| MCC | 0.2203 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 2053.6 (0.49 ms/张) |
| End-to-End FPS | 186.8 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.7273 | 0.5385 | 0.6667 | 0.6957 | 0.6087 | 0.6591 | 0.6609 | 66 |
| malignant | 0.4375 | 0.7791 | 0.4242 | 0.4308 | 0.7882 | 0.6326 | 0.4069 | 32 |
| normal | 0.2000 | 0.9082 | 0.3077 | 0.2424 | 0.8476 | 0.5724 | 0.2418 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=48 FP=24 TN=28 FN=18
- **malignant**: TP=14 FP=19 TN=67 FN=18
- **normal**: TP=4 FP=9 TN=89 FN=16

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 48 | 13 | 5 |
| malignant | 14 | 14 | 4 |
| normal | 10 | 6 | 4 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.6667    0.7273    0.6957        66
   malignant     0.4242    0.4375    0.4308        32
      normal     0.3077    0.2000    0.2424        20

    accuracy                         0.5593       118
   macro avg     0.4662    0.4549    0.4563       118
weighted avg     0.5401    0.5593    0.5470       118

```