# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/repvit_m1_seed44/repvit_m1/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.4372 |
| Accuracy | 0.5776 |
| Macro-AUROC (OvR) | 0.5925 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.5366 |
| Balanced Acc | 0.4398 |
| Macro-Sensitivity | 0.4398 |
| Macro-Specificity | 0.7199 |
| Macro-Precision | 0.5349 |
| AUC-PR (Macro) | 0.4705 |
| Cohen's Kappa | 0.1857 |
| MCC | 0.1994 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 2016.4 (0.50 ms/张) |
| End-to-End FPS | 177.8 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8000 | 0.3333 | 0.6047 | 0.6887 | 0.5667 | 0.5460 | 0.5996 | 65 |
| malignant | 0.4194 | 0.8471 | 0.5000 | 0.4561 | 0.8000 | 0.5696 | 0.4983 | 31 |
| normal | 0.1000 | 0.9792 | 0.5000 | 0.1667 | 0.8393 | 0.6620 | 0.3135 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=52 FP=34 TN=17 FN=13
- **malignant**: TP=13 FP=13 TN=72 FN=18
- **normal**: TP=2 FP=2 TN=94 FN=18

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 52 | 11 | 2 |
| malignant | 18 | 13 | 0 |
| normal | 16 | 2 | 2 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.6047    0.8000    0.6887        65
   malignant     0.5000    0.4194    0.4561        31
      normal     0.5000    0.1000    0.1667        20

    accuracy                         0.5776       116
   macro avg     0.5349    0.4398    0.4372       116
weighted avg     0.5586    0.5776    0.5366       116

```