# convnext_v2b 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2b_seed44/convnext_v2b/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9617 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9953 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9618 |
| Balanced Acc | 0.9623 |
| Macro-Sensitivity | 0.9623 |
| Macro-Specificity | 0.9810 |
| Macro-Precision | 0.9613 |
| AUC-PR (Macro) | 0.9913 |
| Cohen's Kappa | 0.9426 |
| MCC | 0.9428 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.93G |
| Params | 28.22M |
| Pure-Inference FPS | 631.1 (1.58 ms/张) |
| End-to-End FPS | 205.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9864 | 0.9751 | 0.9584 | 0.9680 | 0.9940 | 0.9896 | 208 |
| Malignant_lesions | 0.9602 | 0.9775 | 0.9494 | 0.9548 | 0.9824 | 0.9930 | 0.9869 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9792 | 0.9594 | 0.9717 | 0.9921 | 0.9987 | 0.9974 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=5 TN=363 FN=12
- **Malignant_lesions**: TP=169 FP=9 TN=391 FN=7
- **body_of_stomach_normal**: TP=189 FP=8 TN=376 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 7 | 5 |
| Malignant_lesions | 4 | 169 | 3 |
| body_of_stomach_normal | 1 | 2 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9751    0.9423    0.9584       208
     Malignant_lesions     0.9494    0.9602    0.9548       176
body_of_stomach_normal     0.9594    0.9844    0.9717       192

              accuracy                         0.9618       576
             macro avg     0.9613    0.9623    0.9617       576
          weighted avg     0.9620    0.9618    0.9618       576

```