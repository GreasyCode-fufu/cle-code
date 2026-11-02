# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_svd_seed42/convnext_svd/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8343 |
| Accuracy | 0.8707 |
| Macro-AUROC (OvR) | 0.9544 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8682 |
| Balanced Acc | 0.8201 |
| Macro-Sensitivity | 0.8201 |
| Macro-Specificity | 0.9221 |
| Macro-Precision | 0.8521 |
| AUC-PR (Macro) | 0.9234 |
| Cohen's Kappa | 0.7734 |
| MCC | 0.7756 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 351.7 (2.84 ms/张) |
| End-to-End FPS | 123.8 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9538 | 0.8431 | 0.8857 | 0.9185 | 0.9348 | 0.9532 | 0.9486 | 65 |
| malignant | 0.8065 | 0.9647 | 0.8929 | 0.8475 | 0.9318 | 0.9662 | 0.9431 | 31 |
| normal | 0.7000 | 0.9583 | 0.7778 | 0.7368 | 0.9388 | 0.9438 | 0.8784 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=62 FP=8 TN=43 FN=3
- **malignant**: TP=25 FP=3 TN=82 FN=6
- **normal**: TP=14 FP=4 TN=92 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 62 | 3 | 0 |
| malignant | 2 | 25 | 4 |
| normal | 6 | 0 | 14 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8857    0.9538    0.9185        65
   malignant     0.8929    0.8065    0.8475        31
      normal     0.7778    0.7000    0.7368        20

    accuracy                         0.8707       116
   macro avg     0.8521    0.8201    0.8343       116
weighted avg     0.8690    0.8707    0.8682       116

```