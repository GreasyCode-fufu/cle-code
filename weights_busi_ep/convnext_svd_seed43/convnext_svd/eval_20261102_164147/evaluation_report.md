# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_svd_seed43/convnext_svd/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8554 |
| Accuracy | 0.8644 |
| Macro-AUROC (OvR) | 0.9545 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8646 |
| Balanced Acc | 0.8575 |
| Macro-Sensitivity | 0.8575 |
| Macro-Specificity | 0.9217 |
| Macro-Precision | 0.8535 |
| AUC-PR (Macro) | 0.9266 |
| Cohen's Kappa | 0.7691 |
| MCC | 0.7692 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 466.3 (2.14 ms/张) |
| End-to-End FPS | 125.5 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.8654 | 0.8923 | 0.8855 | 0.8491 | 0.9435 | 0.9620 | 66 |
| malignant | 0.8438 | 0.9302 | 0.8182 | 0.8308 | 0.9412 | 0.9546 | 0.8989 | 32 |
| normal | 0.8500 | 0.9694 | 0.8500 | 0.8500 | 0.9694 | 0.9653 | 0.9189 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=7 TN=45 FN=8
- **malignant**: TP=27 FP=6 TN=80 FN=5
- **normal**: TP=17 FP=3 TN=95 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 6 | 2 |
| malignant | 4 | 27 | 1 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8923    0.8788    0.8855        66
   malignant     0.8182    0.8438    0.8308        32
      normal     0.8500    0.8500    0.8500        20

    accuracy                         0.8644       118
   macro avg     0.8535    0.8575    0.8554       118
weighted avg     0.8650    0.8644    0.8646       118

```