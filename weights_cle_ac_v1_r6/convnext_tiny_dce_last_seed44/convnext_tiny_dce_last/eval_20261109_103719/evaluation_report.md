# convnext_tiny_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_dce_last_seed44/convnext_tiny_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9576 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9581 |
| Balanced Acc | 0.9563 |
| Macro-Sensitivity | 0.9563 |
| Macro-Specificity | 0.9788 |
| Macro-Precision | 0.9601 |
| AUC-PR (Macro) | 0.9925 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9378 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1674.8 (0.60 ms/张) |
| End-to-End FPS | 8.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9620 | 0.9355 | 0.9553 | 0.9861 | 0.9948 | 0.9916 | 208 |
| Malignant_lesions | 0.9034 | 0.9900 | 0.9755 | 0.9381 | 0.9588 | 0.9946 | 0.9880 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9844 | 0.9694 | 0.9794 | 0.9947 | 0.9989 | 0.9979 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=14 TN=354 FN=5
- **Malignant_lesions**: TP=159 FP=4 TN=396 FN=17
- **body_of_stomach_normal**: TP=190 FP=6 TN=378 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 4 | 1 |
| Malignant_lesions | 12 | 159 | 5 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9355    0.9760    0.9553       208
     Malignant_lesions     0.9755    0.9034    0.9381       176
body_of_stomach_normal     0.9694    0.9896    0.9794       192

              accuracy                         0.9583       576
             macro avg     0.9601    0.9563    0.9576       576
          weighted avg     0.9590    0.9583    0.9581       576

```