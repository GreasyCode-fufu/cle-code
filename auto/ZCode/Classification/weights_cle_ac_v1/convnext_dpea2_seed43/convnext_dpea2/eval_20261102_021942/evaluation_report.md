# convnext_dpea2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dpea2_seed43/convnext_dpea2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9594 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9955 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9599 |
| Balanced Acc | 0.9585 |
| Macro-Sensitivity | 0.9585 |
| Macro-Specificity | 0.9798 |
| Macro-Precision | 0.9610 |
| AUC-PR (Macro) | 0.9913 |
| Cohen's Kappa | 0.9399 |
| MCC | 0.9402 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 28.12M |
| Pure-Inference FPS | 1004.4 (1.00 ms/张) |
| End-to-End FPS | 240.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9647 | 0.9395 | 0.9551 | 0.9834 | 0.9946 | 0.9906 | 208 |
| Malignant_lesions | 0.9148 | 0.9850 | 0.9641 | 0.9388 | 0.9633 | 0.9928 | 0.9847 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9896 | 0.9794 | 0.9845 | 0.9948 | 0.9992 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=13 TN=355 FN=6
- **Malignant_lesions**: TP=161 FP=6 TN=394 FN=15
- **body_of_stomach_normal**: TP=190 FP=4 TN=380 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 6 | 0 |
| Malignant_lesions | 11 | 161 | 4 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9395    0.9712    0.9551       208
     Malignant_lesions     0.9641    0.9148    0.9388       176
body_of_stomach_normal     0.9794    0.9896    0.9845       192

              accuracy                         0.9601       576
             macro avg     0.9610    0.9585    0.9594       576
          weighted avg     0.9603    0.9601    0.9599       576

```