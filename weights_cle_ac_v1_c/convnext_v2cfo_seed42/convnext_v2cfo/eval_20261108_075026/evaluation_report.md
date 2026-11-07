# convnext_v2cfo 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v2cfo_seed42/convnext_v2cfo/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9583 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9584 |
| Balanced Acc | 0.9575 |
| Macro-Sensitivity | 0.9575 |
| Macro-Specificity | 0.9789 |
| Macro-Precision | 0.9593 |
| AUC-PR (Macro) | 0.9926 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9375 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1661.2 (0.60 ms/张) |
| End-to-End FPS | 267.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9620 | 0.9349 | 0.9504 | 0.9806 | 0.9950 | 0.9922 | 208 |
| Malignant_lesions | 0.9375 | 0.9800 | 0.9538 | 0.9456 | 0.9727 | 0.9942 | 0.9870 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9948 | 0.9894 | 0.9789 | 0.9845 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=14 TN=354 FN=7
- **Malignant_lesions**: TP=165 FP=8 TN=392 FN=11
- **body_of_stomach_normal**: TP=186 FP=2 TN=382 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 10 | 165 | 1 |
| body_of_stomach_normal | 4 | 2 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9349    0.9663    0.9504       208
     Malignant_lesions     0.9538    0.9375    0.9456       176
body_of_stomach_normal     0.9894    0.9688    0.9789       192

              accuracy                         0.9583       576
             macro avg     0.9593    0.9575    0.9583       576
          weighted avg     0.9588    0.9583    0.9584       576

```