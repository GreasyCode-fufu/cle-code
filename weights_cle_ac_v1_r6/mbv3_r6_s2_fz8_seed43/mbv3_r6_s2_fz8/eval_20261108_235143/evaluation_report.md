# mbv3_r6_s2_fz8 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_fz8_seed43/mbv3_r6_s2_fz8/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9533 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9937 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9532 |
| Balanced Acc | 0.9530 |
| Macro-Sensitivity | 0.9530 |
| Macro-Specificity | 0.9763 |
| Macro-Precision | 0.9537 |
| AUC-PR (Macro) | 0.9885 |
| Cohen's Kappa | 0.9295 |
| MCC | 0.9295 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 5129.5 (0.19 ms/张) |
| End-to-End FPS | 266.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9592 | 0.9289 | 0.9356 | 0.9671 | 0.9896 | 0.9831 | 208 |
| Malignant_lesions | 0.9375 | 0.9750 | 0.9429 | 0.9402 | 0.9726 | 0.9922 | 0.9839 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9948 | 0.9895 | 0.9843 | 0.9896 | 0.9993 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=15 TN=353 FN=12
- **Malignant_lesions**: TP=165 FP=10 TN=390 FN=11
- **body_of_stomach_normal**: TP=188 FP=2 TN=382 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 10 | 2 |
| Malignant_lesions | 11 | 165 | 0 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9289    0.9423    0.9356       208
     Malignant_lesions     0.9429    0.9375    0.9402       176
body_of_stomach_normal     0.9895    0.9792    0.9843       192

              accuracy                         0.9531       576
             macro avg     0.9537    0.9530    0.9533       576
          weighted avg     0.9534    0.9531    0.9532       576

```