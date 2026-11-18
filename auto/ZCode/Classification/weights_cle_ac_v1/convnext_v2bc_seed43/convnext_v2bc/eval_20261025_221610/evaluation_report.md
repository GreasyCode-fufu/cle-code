# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2bc_seed43/convnext_v2bc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9612 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9616 |
| Balanced Acc | 0.9607 |
| Macro-Sensitivity | 0.9607 |
| Macro-Specificity | 0.9808 |
| Macro-Precision | 0.9620 |
| AUC-PR (Macro) | 0.9939 |
| Cohen's Kappa | 0.9425 |
| MCC | 0.9427 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.67M |
| Pure-Inference FPS | 811.1 (1.23 ms/张) |
| End-to-End FPS | 222.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9755 | 0.9571 | 0.9617 | 0.9809 | 0.9953 | 0.9931 | 208 |
| Malignant_lesions | 0.9261 | 0.9850 | 0.9645 | 0.9449 | 0.9681 | 0.9960 | 0.9902 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9818 | 0.9645 | 0.9769 | 0.9947 | 0.9991 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=9 TN=359 FN=7
- **Malignant_lesions**: TP=163 FP=6 TN=394 FN=13
- **body_of_stomach_normal**: TP=190 FP=7 TN=377 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 7 | 163 | 6 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9571    0.9663    0.9617       208
     Malignant_lesions     0.9645    0.9261    0.9449       176
body_of_stomach_normal     0.9645    0.9896    0.9769       192

              accuracy                         0.9618       576
             macro avg     0.9620    0.9607    0.9612       576
          weighted avg     0.9618    0.9618    0.9616       576

```