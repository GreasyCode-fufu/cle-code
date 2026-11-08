# convnext_v5grn4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5grn4_seed43/convnext_v5grn4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9617 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9972 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9620 |
| Balanced Acc | 0.9629 |
| Macro-Sensitivity | 0.9629 |
| Macro-Specificity | 0.9812 |
| Macro-Precision | 0.9614 |
| AUC-PR (Macro) | 0.9947 |
| Cohen's Kappa | 0.9427 |
| MCC | 0.9431 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1630.2 (0.61 ms/张) |
| End-to-End FPS | 267.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9864 | 0.9750 | 0.9559 | 0.9654 | 0.9958 | 0.9931 | 208 |
| Malignant_lesions | 0.9773 | 0.9625 | 0.9198 | 0.9477 | 0.9897 | 0.9964 | 0.9922 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9948 | 0.9894 | 0.9816 | 0.9871 | 0.9994 | 0.9988 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=5 TN=363 FN=13
- **Malignant_lesions**: TP=172 FP=15 TN=385 FN=4
- **body_of_stomach_normal**: TP=187 FP=2 TN=382 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 12 | 1 |
| Malignant_lesions | 3 | 172 | 1 |
| body_of_stomach_normal | 2 | 3 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9750    0.9375    0.9559       208
     Malignant_lesions     0.9198    0.9773    0.9477       176
body_of_stomach_normal     0.9894    0.9740    0.9816       192

              accuracy                         0.9618       576
             macro avg     0.9614    0.9629    0.9617       576
          weighted avg     0.9629    0.9618    0.9620       576

```