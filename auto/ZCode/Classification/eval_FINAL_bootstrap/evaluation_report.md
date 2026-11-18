# convnext_fgc_dsd 评估报告

- 权重: `weights_cle_ac_v1/convnext_fgc_dsd_seed45/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9704 |
| Accuracy | 0.9705 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9705 |
| Balanced Acc | 0.9700 |
| Macro-Sensitivity | 0.9700 |
| Macro-Specificity | 0.9851 |
| Macro-Precision | 0.9708 |
| AUC-PR (Macro) | 0.9932 |
| Cohen's Kappa | 0.9556 |
| MCC | 0.9556 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 1714.3 (0.58 ms/张) |
| End-to-End FPS | 232.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9783 | 0.9619 | 0.9665 | 0.9836 | 0.9941 | 0.9904 | 208 |
| Malignant_lesions | 0.9545 | 0.9875 | 0.9711 | 0.9628 | 0.9801 | 0.9953 | 0.9911 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9896 | 0.9793 | 0.9818 | 0.9922 | 0.9990 | 0.9980 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=8 TN=360 FN=6
- **Malignant_lesions**: TP=168 FP=5 TN=395 FN=8
- **body_of_stomach_normal**: TP=189 FP=4 TN=380 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 5 | 168 | 3 |
| body_of_stomach_normal | 3 | 0 | 189 |

## 核心指标 95% Bootstrap CI

| 指标 | 95% CI |
|---|---|
| macro_f1 | [0.9548, 0.9840] |
| accuracy | [0.9549, 0.9844] |
| auc_roc_macro | [0.9928, 0.9987] |
| macro_sensitivity | [0.9549, 0.9838] |
| macro_specificity | [0.9775, 0.9921] |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9619    0.9712    0.9665       208
     Malignant_lesions     0.9711    0.9545    0.9628       176
body_of_stomach_normal     0.9793    0.9844    0.9818       192

              accuracy                         0.9705       576
             macro avg     0.9708    0.9700    0.9704       576
          weighted avg     0.9705    0.9705    0.9705       576

```