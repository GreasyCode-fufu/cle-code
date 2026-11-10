# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/convnext_tiny_seed44/convnext_tiny/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8757 |
| Accuracy | 0.8966 |
| Macro-AUROC (OvR) | 0.9561 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8975 |
| Balanced Acc | 0.8873 |
| Macro-Sensitivity | 0.8873 |
| Macro-Specificity | 0.9473 |
| Macro-Precision | 0.8672 |
| AUC-PR (Macro) | 0.9257 |
| Cohen's Kappa | 0.8254 |
| MCC | 0.8261 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 821.1 (1.22 ms/张) |
| End-to-End FPS | 153.4 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9231 | 0.9412 | 0.9524 | 0.9375 | 0.9057 | 0.9641 | 0.9565 | 65 |
| malignant | 0.8387 | 0.9529 | 0.8667 | 0.8525 | 0.9419 | 0.9526 | 0.9198 | 31 |
| normal | 0.9000 | 0.9479 | 0.7826 | 0.8372 | 0.9785 | 0.9516 | 0.9009 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=3 TN=48 FN=5
- **malignant**: TP=26 FP=4 TN=81 FN=5
- **normal**: TP=18 FP=5 TN=91 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 4 | 1 |
| malignant | 1 | 26 | 4 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9524    0.9231    0.9375        65
   malignant     0.8667    0.8387    0.8525        31
      normal     0.7826    0.9000    0.8372        20

    accuracy                         0.8966       116
   macro avg     0.8672    0.8873    0.8757       116
weighted avg     0.9002    0.8966    0.8975       116

```