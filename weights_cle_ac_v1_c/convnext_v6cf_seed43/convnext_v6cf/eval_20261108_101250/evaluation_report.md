# convnext_v6cf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v6cf_seed43/convnext_v6cf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9407 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9939 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9415 |
| Balanced Acc | 0.9420 |
| Macro-Sensitivity | 0.9420 |
| Macro-Specificity | 0.9711 |
| Macro-Precision | 0.9410 |
| AUC-PR (Macro) | 0.9883 |
| Cohen's Kappa | 0.9114 |
| MCC | 0.9122 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 5.79G |
| FLOPs | 11.58G |
| Params | 30.61M |
| Pure-Inference FPS | 399.7 (2.50 ms/张) |
| End-to-End FPS | 172.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9135 | 0.9810 | 0.9645 | 0.9383 | 0.9525 | 0.9918 | 0.9875 | 208 |
| Malignant_lesions | 0.9489 | 0.9400 | 0.8743 | 0.9101 | 0.9766 | 0.9905 | 0.9786 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9922 | 0.9840 | 0.9737 | 0.9820 | 0.9993 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=7 TN=361 FN=18
- **Malignant_lesions**: TP=167 FP=24 TN=376 FN=9
- **body_of_stomach_normal**: TP=185 FP=3 TN=381 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 18 | 0 |
| Malignant_lesions | 6 | 167 | 3 |
| body_of_stomach_normal | 1 | 6 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9645    0.9135    0.9383       208
     Malignant_lesions     0.8743    0.9489    0.9101       176
body_of_stomach_normal     0.9840    0.9635    0.9737       192

              accuracy                         0.9410       576
             macro avg     0.9410    0.9420    0.9407       576
          weighted avg     0.9435    0.9410    0.9415       576

```