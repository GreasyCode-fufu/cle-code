# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r3/convnext_v4e_seed43/convnext_v4e/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8349 |
| Accuracy | 0.8475 |
| Macro-AUROC (OvR) | 0.9410 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8500 |
| Balanced Acc | 0.8706 |
| Macro-Sensitivity | 0.8706 |
| Macro-Specificity | 0.9264 |
| Macro-Precision | 0.8151 |
| AUC-PR (Macro) | 0.8951 |
| Cohen's Kappa | 0.7511 |
| MCC | 0.7576 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 373.0 (2.68 ms/张) |
| End-to-End FPS | 130.0 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8182 | 0.9423 | 0.9474 | 0.8780 | 0.8033 | 0.9104 | 0.9434 | 66 |
| malignant | 0.8438 | 0.9186 | 0.7941 | 0.8182 | 0.9405 | 0.9459 | 0.8613 | 32 |
| normal | 0.9500 | 0.9184 | 0.7037 | 0.8085 | 0.9890 | 0.9668 | 0.8808 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=54 FP=3 TN=49 FN=12
- **malignant**: TP=27 FP=7 TN=79 FN=5
- **normal**: TP=19 FP=8 TN=90 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 54 | 7 | 5 |
| malignant | 2 | 27 | 3 |
| normal | 1 | 0 | 19 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9474    0.8182    0.8780        66
   malignant     0.7941    0.8438    0.8182        32
      normal     0.7037    0.9500    0.8085        20

    accuracy                         0.8475       118
   macro avg     0.8151    0.8706    0.8349       118
weighted avg     0.8645    0.8475    0.8500       118

```