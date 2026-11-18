# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/ghostnet_seed44/ghostnet/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8486 |
| Accuracy | 0.8707 |
| Macro-AUROC (OvR) | 0.9554 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8694 |
| Balanced Acc | 0.8376 |
| Macro-Sensitivity | 0.8376 |
| Macro-Specificity | 0.9194 |
| Macro-Precision | 0.8648 |
| AUC-PR (Macro) | 0.9164 |
| Cohen's Kappa | 0.7740 |
| MCC | 0.7768 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 1369.7 (0.73 ms/张) |
| End-to-End FPS | 170.9 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.8235 | 0.8714 | 0.9037 | 0.9130 | 0.9460 | 0.9558 | 65 |
| malignant | 0.7742 | 0.9765 | 0.9231 | 0.8421 | 0.9222 | 0.9677 | 0.9092 | 31 |
| normal | 0.8000 | 0.9583 | 0.8000 | 0.8000 | 0.9583 | 0.9526 | 0.8841 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=9 TN=42 FN=4
- **malignant**: TP=24 FP=2 TN=83 FN=7
- **normal**: TP=16 FP=4 TN=92 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 2 | 2 |
| malignant | 5 | 24 | 2 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8714    0.9385    0.9037        65
   malignant     0.9231    0.7742    0.8421        31
      normal     0.8000    0.8000    0.8000        20

    accuracy                         0.8707       116
   macro avg     0.8648    0.8376    0.8486       116
weighted avg     0.8729    0.8707    0.8694       116

```