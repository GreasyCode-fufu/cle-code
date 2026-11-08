# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v2m_seed44/convnext_v2m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9582 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9945 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9585 |
| Balanced Acc | 0.9583 |
| Macro-Sensitivity | 0.9583 |
| Macro-Specificity | 0.9792 |
| Macro-Precision | 0.9583 |
| AUC-PR (Macro) | 0.9902 |
| Cohen's Kappa | 0.9374 |
| MCC | 0.9375 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1654.2 (0.60 ms/张) |
| End-to-End FPS | 271.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9728 | 0.9522 | 0.9544 | 0.9755 | 0.9915 | 0.9880 | 208 |
| Malignant_lesions | 0.9545 | 0.9700 | 0.9333 | 0.9438 | 0.9798 | 0.9931 | 0.9849 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9948 | 0.9893 | 0.9763 | 0.9820 | 0.9989 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=10 TN=358 FN=9
- **Malignant_lesions**: TP=168 FP=12 TN=388 FN=8
- **body_of_stomach_normal**: TP=185 FP=2 TN=382 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 7 | 168 | 1 |
| body_of_stomach_normal | 3 | 4 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9522    0.9567    0.9544       208
     Malignant_lesions     0.9333    0.9545    0.9438       176
body_of_stomach_normal     0.9893    0.9635    0.9763       192

              accuracy                         0.9583       576
             macro avg     0.9583    0.9583    0.9582       576
          weighted avg     0.9588    0.9583    0.9585       576

```