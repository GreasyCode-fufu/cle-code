# mbv3_r6_s2x_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2x_fz16_seed44/mbv3_r6_s2x_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9620 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9960 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9618 |
| Balanced Acc | 0.9620 |
| Macro-Sensitivity | 0.9620 |
| Macro-Specificity | 0.9808 |
| Macro-Precision | 0.9619 |
| AUC-PR (Macro) | 0.9927 |
| Cohen's Kappa | 0.9426 |
| MCC | 0.9426 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.10M |
| FLOPs | 464.20M |
| Params | 3.39M |
| Pure-Inference FPS | 4435.2 (0.23 ms/张) |
| End-to-End FPS | 207.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9728 | 0.9519 | 0.9519 | 0.9728 | 0.9933 | 0.9901 | 208 |
| Malignant_lesions | 0.9602 | 0.9800 | 0.9548 | 0.9575 | 0.9825 | 0.9957 | 0.9898 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9896 | 0.9791 | 0.9765 | 0.9870 | 0.9990 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=10 TN=358 FN=10
- **Malignant_lesions**: TP=169 FP=8 TN=392 FN=7
- **body_of_stomach_normal**: TP=187 FP=4 TN=380 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 6 | 169 | 1 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9519    0.9519    0.9519       208
     Malignant_lesions     0.9548    0.9602    0.9575       176
body_of_stomach_normal     0.9791    0.9740    0.9765       192

              accuracy                         0.9618       576
             macro avg     0.9619    0.9620    0.9620       576
          weighted avg     0.9618    0.9618    0.9618       576

```