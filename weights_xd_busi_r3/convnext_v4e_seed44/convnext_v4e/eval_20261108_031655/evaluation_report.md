# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r3/convnext_v4e_seed44/convnext_v4e/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8819 |
| Accuracy | 0.9052 |
| Macro-AUROC (OvR) | 0.9646 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9051 |
| Balanced Acc | 0.8868 |
| Macro-Sensitivity | 0.8868 |
| Macro-Specificity | 0.9487 |
| Macro-Precision | 0.8826 |
| AUC-PR (Macro) | 0.9211 |
| Cohen's Kappa | 0.8378 |
| MCC | 0.8392 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 474.0 (2.11 ms/张) |
| End-to-End FPS | 133.0 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9538 | 0.9216 | 0.9394 | 0.9466 | 0.9400 | 0.9674 | 0.9706 | 65 |
| malignant | 0.8065 | 0.9765 | 0.9259 | 0.8621 | 0.9326 | 0.9613 | 0.9263 | 31 |
| normal | 0.9000 | 0.9479 | 0.7826 | 0.8372 | 0.9785 | 0.9651 | 0.8663 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=62 FP=4 TN=47 FN=3
- **malignant**: TP=25 FP=2 TN=83 FN=6
- **normal**: TP=18 FP=5 TN=91 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 62 | 2 | 1 |
| malignant | 2 | 25 | 4 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9394    0.9538    0.9466        65
   malignant     0.9259    0.8065    0.8621        31
      normal     0.7826    0.9000    0.8372        20

    accuracy                         0.9052       116
   macro avg     0.8826    0.8868    0.8819       116
weighted avg     0.9088    0.9052    0.9051       116

```