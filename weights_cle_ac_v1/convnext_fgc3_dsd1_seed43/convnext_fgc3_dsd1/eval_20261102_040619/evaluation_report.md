# convnext_fgc3_dsd1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc3_dsd1_seed43/convnext_fgc3_dsd1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9319 |
| Accuracy | 0.9323 |
| Macro-AUROC (OvR) | 0.9896 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9322 |
| Balanced Acc | 0.9297 |
| Macro-Sensitivity | 0.9297 |
| Macro-Specificity | 0.9654 |
| Macro-Precision | 0.9367 |
| AUC-PR (Macro) | 0.9813 |
| Cohen's Kappa | 0.8979 |
| MCC | 0.8992 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.39G |
| FLOPs | 8.78G |
| Params | 27.62M |
| Pure-Inference FPS | 855.9 (1.17 ms/张) |
| End-to-End FPS | 227.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9239 | 0.8761 | 0.9124 | 0.9714 | 0.9854 | 0.9778 | 208 |
| Malignant_lesions | 0.8580 | 0.9800 | 0.9497 | 0.9015 | 0.9400 | 0.9849 | 0.9692 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9984 | 0.9968 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=28 TN=340 FN=10
- **Malignant_lesions**: TP=151 FP=8 TN=392 FN=25
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 8 | 2 |
| Malignant_lesions | 24 | 151 | 1 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8761    0.9519    0.9124       208
     Malignant_lesions     0.9497    0.8580    0.9015       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9323       576
             macro avg     0.9367    0.9297    0.9319       576
          weighted avg     0.9347    0.9323    0.9322       576

```