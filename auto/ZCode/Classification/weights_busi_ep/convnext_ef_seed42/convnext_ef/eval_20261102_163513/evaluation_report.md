# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_ef_seed42/convnext_ef/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8692 |
| Accuracy | 0.8879 |
| Macro-AUROC (OvR) | 0.9567 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8881 |
| Balanced Acc | 0.8765 |
| Macro-Sensitivity | 0.8765 |
| Macro-Specificity | 0.9382 |
| Macro-Precision | 0.8662 |
| AUC-PR (Macro) | 0.9226 |
| Cohen's Kappa | 0.8092 |
| MCC | 0.8101 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 330.8 (3.02 ms/张) |
| End-to-End FPS | 120.4 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9231 | 0.9020 | 0.9231 | 0.9231 | 0.9020 | 0.9653 | 0.9401 | 65 |
| malignant | 0.8065 | 0.9647 | 0.8929 | 0.8475 | 0.9318 | 0.9651 | 0.9344 | 31 |
| normal | 0.9000 | 0.9479 | 0.7826 | 0.8372 | 0.9785 | 0.9396 | 0.8934 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=5 TN=46 FN=5
- **malignant**: TP=25 FP=3 TN=82 FN=6
- **normal**: TP=18 FP=5 TN=91 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 3 | 2 |
| malignant | 3 | 25 | 3 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9231    0.9231    0.9231        65
   malignant     0.8929    0.8065    0.8475        31
      normal     0.7826    0.9000    0.8372        20

    accuracy                         0.8879       116
   macro avg     0.8662    0.8765    0.8692       116
weighted avg     0.8908    0.8879    0.8881       116

```