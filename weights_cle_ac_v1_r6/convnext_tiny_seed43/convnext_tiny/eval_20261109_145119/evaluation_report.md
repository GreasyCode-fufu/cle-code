# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_seed43/convnext_tiny/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9373 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9918 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9375 |
| Balanced Acc | 0.9363 |
| Macro-Sensitivity | 0.9363 |
| Macro-Specificity | 0.9684 |
| Macro-Precision | 0.9387 |
| AUC-PR (Macro) | 0.9855 |
| Cohen's Kappa | 0.9059 |
| MCC | 0.9062 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 491.2 (2.04 ms/张) |
| End-to-End FPS | 189.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9484 | 0.9120 | 0.9292 | 0.9694 | 0.9898 | 0.9815 | 208 |
| Malignant_lesions | 0.9034 | 0.9750 | 0.9408 | 0.9217 | 0.9582 | 0.9880 | 0.9796 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9818 | 0.9634 | 0.9608 | 0.9792 | 0.9976 | 0.9954 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=19 TN=349 FN=11
- **Malignant_lesions**: TP=159 FP=10 TN=390 FN=17
- **body_of_stomach_normal**: TP=184 FP=7 TN=377 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 8 | 3 |
| Malignant_lesions | 13 | 159 | 4 |
| body_of_stomach_normal | 6 | 2 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9120    0.9471    0.9292       208
     Malignant_lesions     0.9408    0.9034    0.9217       176
body_of_stomach_normal     0.9634    0.9583    0.9608       192

              accuracy                         0.9375       576
             macro avg     0.9387    0.9363    0.9373       576
          weighted avg     0.9379    0.9375    0.9375       576

```