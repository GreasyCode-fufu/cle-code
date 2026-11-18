# convnext_v2cema 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v2cema_seed43/convnext_v2cema/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9583 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9956 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9584 |
| Balanced Acc | 0.9574 |
| Macro-Sensitivity | 0.9574 |
| Macro-Specificity | 0.9789 |
| Macro-Precision | 0.9596 |
| AUC-PR (Macro) | 0.9907 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9375 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1656.8 (0.60 ms/张) |
| End-to-End FPS | 270.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9592 | 0.9306 | 0.9481 | 0.9806 | 0.9939 | 0.9918 | 208 |
| Malignant_lesions | 0.9318 | 0.9800 | 0.9535 | 0.9425 | 0.9703 | 0.9930 | 0.9807 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9974 | 0.9947 | 0.9842 | 0.9871 | 0.9998 | 0.9995 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=15 TN=353 FN=7
- **Malignant_lesions**: TP=164 FP=8 TN=392 FN=12
- **body_of_stomach_normal**: TP=187 FP=1 TN=383 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 7 | 0 |
| Malignant_lesions | 11 | 164 | 1 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9306    0.9663    0.9481       208
     Malignant_lesions     0.9535    0.9318    0.9425       176
body_of_stomach_normal     0.9947    0.9740    0.9842       192

              accuracy                         0.9583       576
             macro avg     0.9596    0.9574    0.9583       576
          weighted avg     0.9589    0.9583    0.9584       576

```