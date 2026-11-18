# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_ef_seed42/convnext_ef/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8194 |
| Accuracy | 0.8390 |
| Macro-AUROC (OvR) | 0.9390 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8411 |
| Balanced Acc | 0.8432 |
| Macro-Sensitivity | 0.8432 |
| Macro-Specificity | 0.9175 |
| Macro-Precision | 0.8056 |
| AUC-PR (Macro) | 0.8828 |
| Cohen's Kappa | 0.7326 |
| MCC | 0.7356 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 318.8 (3.14 ms/张) |
| End-to-End FPS | 122.2 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8485 | 0.9038 | 0.9180 | 0.8819 | 0.8246 | 0.9191 | 0.9469 | 66 |
| malignant | 0.7812 | 0.9302 | 0.8065 | 0.7937 | 0.9195 | 0.9320 | 0.8395 | 32 |
| normal | 0.9000 | 0.9184 | 0.6923 | 0.7826 | 0.9783 | 0.9658 | 0.8621 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=5 TN=47 FN=10
- **malignant**: TP=25 FP=6 TN=80 FN=7
- **normal**: TP=18 FP=8 TN=90 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 5 | 5 |
| malignant | 4 | 25 | 3 |
| normal | 1 | 1 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9180    0.8485    0.8819        66
   malignant     0.8065    0.7812    0.7937        32
      normal     0.6923    0.9000    0.7826        20

    accuracy                         0.8390       118
   macro avg     0.8056    0.8432    0.8194       118
weighted avg     0.8495    0.8390    0.8411       118

```