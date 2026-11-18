# convnext_v3f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v3f_seed42/convnext_v3f/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9614 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9969 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9617 |
| Balanced Acc | 0.9605 |
| Macro-Sensitivity | 0.9605 |
| Macro-Specificity | 0.9806 |
| Macro-Precision | 0.9629 |
| AUC-PR (Macro) | 0.9942 |
| Cohen's Kappa | 0.9425 |
| MCC | 0.9428 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.65G |
| FLOPs | 9.31G |
| Params | 28.73M |
| Pure-Inference FPS | 324.5 (3.08 ms/张) |
| End-to-End FPS | 156.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9647 | 0.9393 | 0.9526 | 0.9807 | 0.9951 | 0.9923 | 208 |
| Malignant_lesions | 0.9205 | 0.9875 | 0.9701 | 0.9446 | 0.9658 | 0.9960 | 0.9912 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9896 | 0.9795 | 0.9871 | 0.9974 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=13 TN=355 FN=7
- **Malignant_lesions**: TP=162 FP=5 TN=395 FN=14
- **body_of_stomach_normal**: TP=191 FP=4 TN=380 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 12 | 162 | 2 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9393    0.9663    0.9526       208
     Malignant_lesions     0.9701    0.9205    0.9446       176
body_of_stomach_normal     0.9795    0.9948    0.9871       192

              accuracy                         0.9618       576
             macro avg     0.9629    0.9605    0.9614       576
          weighted avg     0.9621    0.9618    0.9617       576

```