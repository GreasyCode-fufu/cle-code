# mbv3_r6_s2_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_fz16_seed45/mbv3_r6_s2_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9620 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9937 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9618 |
| Balanced Acc | 0.9619 |
| Macro-Sensitivity | 0.9619 |
| Macro-Specificity | 0.9808 |
| Macro-Precision | 0.9621 |
| AUC-PR (Macro) | 0.9882 |
| Cohen's Kappa | 0.9426 |
| MCC | 0.9426 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 4508.6 (0.22 ms/张) |
| End-to-End FPS | 110.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9701 | 0.9474 | 0.9496 | 0.9728 | 0.9897 | 0.9842 | 208 |
| Malignant_lesions | 0.9545 | 0.9800 | 0.9545 | 0.9545 | 0.9800 | 0.9927 | 0.9833 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9986 | 0.9971 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=11 TN=357 FN=10
- **Malignant_lesions**: TP=168 FP=8 TN=392 FN=8
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 8 | 2 |
| Malignant_lesions | 7 | 168 | 1 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9474    0.9519    0.9496       208
     Malignant_lesions     0.9545    0.9545    0.9545       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9618       576
             macro avg     0.9621    0.9619    0.9620       576
          weighted avg     0.9619    0.9618    0.9618       576

```