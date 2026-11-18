# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r5/convnext_v6f_seed43/convnext_v6f/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8720 |
| Accuracy | 0.8966 |
| Macro-AUROC (OvR) | 0.9575 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8948 |
| Balanced Acc | 0.8586 |
| Macro-Sensitivity | 0.8586 |
| Macro-Specificity | 0.9360 |
| Macro-Precision | 0.8893 |
| AUC-PR (Macro) | 0.9132 |
| Cohen's Kappa | 0.8190 |
| MCC | 0.8215 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 906.2 (1.10 ms/张) |
| End-to-End FPS | 157.5 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9692 | 0.8627 | 0.9000 | 0.9333 | 0.9565 | 0.9710 | 0.9640 | 65 |
| malignant | 0.8065 | 0.9765 | 0.9259 | 0.8621 | 0.9326 | 0.9609 | 0.9336 | 31 |
| normal | 0.8000 | 0.9688 | 0.8421 | 0.8205 | 0.9588 | 0.9406 | 0.8421 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=63 FP=7 TN=44 FN=2
- **malignant**: TP=25 FP=2 TN=83 FN=6
- **normal**: TP=16 FP=3 TN=93 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 63 | 2 | 0 |
| malignant | 3 | 25 | 3 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9000    0.9692    0.9333        65
   malignant     0.9259    0.8065    0.8621        31
      normal     0.8421    0.8000    0.8205        20

    accuracy                         0.8966       116
   macro avg     0.8893    0.8586    0.8720       116
weighted avg     0.8969    0.8966    0.8948       116

```