# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_tiny_seed44/convnext_tiny/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9615 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9960 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9618 |
| Balanced Acc | 0.9616 |
| Macro-Sensitivity | 0.9616 |
| Macro-Specificity | 0.9809 |
| Macro-Precision | 0.9615 |
| AUC-PR (Macro) | 0.9925 |
| Cohen's Kappa | 0.9426 |
| MCC | 0.9426 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 509.6 (1.96 ms/张) |
| End-to-End FPS | 191.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9755 | 0.9567 | 0.9567 | 0.9755 | 0.9948 | 0.9911 | 208 |
| Malignant_lesions | 0.9489 | 0.9750 | 0.9435 | 0.9462 | 0.9774 | 0.9940 | 0.9880 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=9 TN=359 FN=9
- **Malignant_lesions**: TP=167 FP=10 TN=390 FN=9
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 7 | 167 | 2 |
| body_of_stomach_normal | 2 | 2 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9567    0.9567    0.9567       208
     Malignant_lesions     0.9435    0.9489    0.9462       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9618       576
             macro avg     0.9615    0.9616    0.9615       576
          weighted avg     0.9619    0.9618    0.9618       576

```