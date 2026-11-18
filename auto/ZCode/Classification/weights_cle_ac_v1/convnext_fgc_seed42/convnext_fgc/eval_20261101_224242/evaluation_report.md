# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_seed42/convnext_fgc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9496 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9953 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9499 |
| Balanced Acc | 0.9498 |
| Macro-Sensitivity | 0.9498 |
| Macro-Specificity | 0.9749 |
| Macro-Precision | 0.9497 |
| AUC-PR (Macro) | 0.9914 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9245 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 598.4 (1.67 ms/张) |
| End-to-End FPS | 202.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9674 | 0.9423 | 0.9423 | 0.9674 | 0.9939 | 0.9895 | 208 |
| Malignant_lesions | 0.9489 | 0.9625 | 0.9176 | 0.9330 | 0.9772 | 0.9930 | 0.9864 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9948 | 0.9892 | 0.9735 | 0.9795 | 0.9991 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=12 TN=356 FN=12
- **Malignant_lesions**: TP=167 FP=15 TN=385 FN=9
- **body_of_stomach_normal**: TP=184 FP=2 TN=382 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 11 | 1 |
| Malignant_lesions | 8 | 167 | 1 |
| body_of_stomach_normal | 4 | 4 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9423    0.9423    0.9423       208
     Malignant_lesions     0.9176    0.9489    0.9330       176
body_of_stomach_normal     0.9892    0.9583    0.9735       192

              accuracy                         0.9497       576
             macro avg     0.9497    0.9498    0.9496       576
          weighted avg     0.9504    0.9497    0.9499       576

```