# mbv3_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce16_seed43/mbv3_dce16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9530 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9921 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9531 |
| Balanced Acc | 0.9523 |
| Macro-Sensitivity | 0.9523 |
| Macro-Specificity | 0.9763 |
| Macro-Precision | 0.9539 |
| AUC-PR (Macro) | 0.9844 |
| Cohen's Kappa | 0.9295 |
| MCC | 0.9296 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 3573.1 (0.28 ms/张) |
| End-to-End FPS | 14.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9592 | 0.9296 | 0.9406 | 0.9725 | 0.9883 | 0.9799 | 208 |
| Malignant_lesions | 0.9205 | 0.9800 | 0.9529 | 0.9364 | 0.9655 | 0.9884 | 0.9744 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9896 | 0.9793 | 0.9818 | 0.9922 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=15 TN=353 FN=10
- **Malignant_lesions**: TP=162 FP=8 TN=392 FN=14
- **body_of_stomach_normal**: TP=189 FP=4 TN=380 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 13 | 162 | 1 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9296    0.9519    0.9406       208
     Malignant_lesions     0.9529    0.9205    0.9364       176
body_of_stomach_normal     0.9793    0.9844    0.9818       192

              accuracy                         0.9531       576
             macro avg     0.9539    0.9523    0.9530       576
          weighted avg     0.9533    0.9531    0.9531       576

```