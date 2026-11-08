# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r4/convnext_v2c_seed43/convnext_v2c/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8926 |
| Accuracy | 0.9138 |
| Macro-AUROC (OvR) | 0.9616 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9139 |
| Balanced Acc | 0.8975 |
| Macro-Sensitivity | 0.8975 |
| Macro-Specificity | 0.9547 |
| Macro-Precision | 0.8895 |
| AUC-PR (Macro) | 0.9331 |
| Cohen's Kappa | 0.8530 |
| MCC | 0.8535 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1280.3 (0.78 ms/张) |
| End-to-End FPS | 170.1 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9538 | 0.9412 | 0.9538 | 0.9538 | 0.9412 | 0.9710 | 0.9662 | 65 |
| malignant | 0.8387 | 0.9647 | 0.8966 | 0.8667 | 0.9425 | 0.9689 | 0.9446 | 31 |
| normal | 0.9000 | 0.9583 | 0.8182 | 0.8571 | 0.9787 | 0.9448 | 0.8883 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=62 FP=3 TN=48 FN=3
- **malignant**: TP=26 FP=3 TN=82 FN=5
- **normal**: TP=18 FP=4 TN=92 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 62 | 3 | 0 |
| malignant | 1 | 26 | 4 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9538    0.9538    0.9538        65
   malignant     0.8966    0.8387    0.8667        31
      normal     0.8182    0.9000    0.8571        20

    accuracy                         0.9138       116
   macro avg     0.8895    0.8975    0.8926       116
weighted avg     0.9151    0.9138    0.9139       116

```