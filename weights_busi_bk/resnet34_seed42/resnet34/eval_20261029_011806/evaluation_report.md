# resnet34 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/resnet34_seed42/resnet34/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8511 |
| Accuracy | 0.8707 |
| Macro-AUROC (OvR) | 0.9464 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8705 |
| Balanced Acc | 0.8488 |
| Macro-Sensitivity | 0.8488 |
| Macro-Specificity | 0.9247 |
| Macro-Precision | 0.8535 |
| AUC-PR (Macro) | 0.9243 |
| Cohen's Kappa | 0.7779 |
| MCC | 0.7781 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 1921.8 (0.52 ms/张) |
| End-to-End FPS | 177.6 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9077 | 0.8627 | 0.8939 | 0.9008 | 0.8800 | 0.9454 | 0.9343 | 65 |
| malignant | 0.8387 | 0.9529 | 0.8667 | 0.8525 | 0.9419 | 0.9510 | 0.9283 | 31 |
| normal | 0.8000 | 0.9583 | 0.8000 | 0.8000 | 0.9583 | 0.9427 | 0.9103 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=7 TN=44 FN=6
- **malignant**: TP=26 FP=4 TN=81 FN=5
- **normal**: TP=16 FP=4 TN=92 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 4 | 2 |
| malignant | 3 | 26 | 2 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8939    0.9077    0.9008        65
   malignant     0.8667    0.8387    0.8525        31
      normal     0.8000    0.8000    0.8000        20

    accuracy                         0.8707       116
   macro avg     0.8535    0.8488    0.8511       116
weighted avg     0.8705    0.8707    0.8705       116

```