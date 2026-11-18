# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_ef_seed43/convnext_ef/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8551 |
| Accuracy | 0.8879 |
| Macro-AUROC (OvR) | 0.9605 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8839 |
| Balanced Acc | 0.8366 |
| Macro-Sensitivity | 0.8366 |
| Macro-Specificity | 0.9273 |
| Macro-Precision | 0.8928 |
| AUC-PR (Macro) | 0.9241 |
| Cohen's Kappa | 0.8006 |
| MCC | 0.8104 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 311.0 (3.22 ms/张) |
| End-to-End FPS | 116.3 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 1.0000 | 0.8235 | 0.8784 | 0.9353 | 1.0000 | 0.9704 | 0.9615 | 65 |
| malignant | 0.7097 | 1.0000 | 1.0000 | 0.8302 | 0.9043 | 0.9674 | 0.9428 | 31 |
| normal | 0.8000 | 0.9583 | 0.8000 | 0.8000 | 0.9583 | 0.9437 | 0.8679 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=65 FP=9 TN=42 FN=0
- **malignant**: TP=22 FP=0 TN=85 FN=9
- **normal**: TP=16 FP=4 TN=92 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 65 | 0 | 0 |
| malignant | 5 | 22 | 4 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8784    1.0000    0.9353        65
   malignant     1.0000    0.7097    0.8302        31
      normal     0.8000    0.8000    0.8000        20

    accuracy                         0.8879       116
   macro avg     0.8928    0.8366    0.8551       116
weighted avg     0.8974    0.8879    0.8839       116

```