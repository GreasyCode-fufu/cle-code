# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mobilenet_v3_large_seed43/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8531 |
| Accuracy | 0.8793 |
| Macro-AUROC (OvR) | 0.9522 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8799 |
| Balanced Acc | 0.8598 |
| Macro-Sensitivity | 0.8598 |
| Macro-Specificity | 0.9369 |
| Macro-Precision | 0.8479 |
| AUC-PR (Macro) | 0.9041 |
| Cohen's Kappa | 0.7951 |
| MCC | 0.7954 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 1517.7 (0.66 ms/张) |
| End-to-End FPS | 165.0 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9231 | 0.9216 | 0.9375 | 0.9302 | 0.9038 | 0.9520 | 0.9474 | 65 |
| malignant | 0.8065 | 0.9412 | 0.8333 | 0.8197 | 0.9302 | 0.9552 | 0.8997 | 31 |
| normal | 0.8500 | 0.9479 | 0.7727 | 0.8095 | 0.9681 | 0.9495 | 0.8651 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=4 TN=47 FN=5
- **malignant**: TP=25 FP=5 TN=80 FN=6
- **normal**: TP=17 FP=5 TN=91 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 4 | 1 |
| malignant | 2 | 25 | 4 |
| normal | 2 | 1 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9375    0.9231    0.9302        65
   malignant     0.8333    0.8065    0.8197        31
      normal     0.7727    0.8500    0.8095        20

    accuracy                         0.8793       116
   macro avg     0.8479    0.8598    0.8531       116
weighted avg     0.8813    0.8793    0.8799       116

```