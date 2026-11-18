# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce16_seed45/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9621 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9941 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9619 |
| Balanced Acc | 0.9622 |
| Macro-Sensitivity | 0.9622 |
| Macro-Specificity | 0.9808 |
| Macro-Precision | 0.9622 |
| AUC-PR (Macro) | 0.9886 |
| Cohen's Kappa | 0.9426 |
| MCC | 0.9427 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 5625.6 (0.18 ms/张) |
| End-to-End FPS | 14.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9701 | 0.9474 | 0.9496 | 0.9728 | 0.9906 | 0.9870 | 208 |
| Malignant_lesions | 0.9659 | 0.9750 | 0.9444 | 0.9551 | 0.9848 | 0.9923 | 0.9797 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9974 | 0.9947 | 0.9815 | 0.9846 | 0.9996 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=11 TN=357 FN=10
- **Malignant_lesions**: TP=170 FP=10 TN=390 FN=6
- **body_of_stomach_normal**: TP=186 FP=1 TN=383 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 9 | 1 |
| Malignant_lesions | 6 | 170 | 0 |
| body_of_stomach_normal | 5 | 1 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9474    0.9519    0.9496       208
     Malignant_lesions     0.9444    0.9659    0.9551       176
body_of_stomach_normal     0.9947    0.9688    0.9815       192

              accuracy                         0.9618       576
             macro avg     0.9622    0.9622    0.9621       576
          weighted avg     0.9622    0.9618    0.9619       576

```