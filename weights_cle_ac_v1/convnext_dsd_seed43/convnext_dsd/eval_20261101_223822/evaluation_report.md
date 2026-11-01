# convnext_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dsd_seed43/convnext_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9474 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9952 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9477 |
| Balanced Acc | 0.9472 |
| Macro-Sensitivity | 0.9472 |
| Macro-Specificity | 0.9738 |
| Macro-Precision | 0.9479 |
| AUC-PR (Macro) | 0.9904 |
| Cohen's Kappa | 0.9217 |
| MCC | 0.9217 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 27.78M |
| Pure-Inference FPS | 594.9 (1.68 ms/张) |
| End-to-End FPS | 201.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9647 | 0.9378 | 0.9400 | 0.9673 | 0.9941 | 0.9909 | 208 |
| Malignant_lesions | 0.9148 | 0.9750 | 0.9415 | 0.9280 | 0.9630 | 0.9920 | 0.9816 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9818 | 0.9643 | 0.9742 | 0.9921 | 0.9993 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=13 TN=355 FN=12
- **Malignant_lesions**: TP=161 FP=10 TN=390 FN=15
- **body_of_stomach_normal**: TP=189 FP=7 TN=377 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 9 | 3 |
| Malignant_lesions | 11 | 161 | 4 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9378    0.9423    0.9400       208
     Malignant_lesions     0.9415    0.9148    0.9280       176
body_of_stomach_normal     0.9643    0.9844    0.9742       192

              accuracy                         0.9479       576
             macro avg     0.9479    0.9472    0.9474       576
          weighted avg     0.9478    0.9479    0.9477       576

```