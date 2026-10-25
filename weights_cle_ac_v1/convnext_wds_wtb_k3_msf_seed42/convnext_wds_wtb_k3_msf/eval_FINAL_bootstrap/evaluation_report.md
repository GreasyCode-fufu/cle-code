# convnext_wds_wtb_k3_msf 评估报告

- 权重: `weights_cle_ac_v1/convnext_wds_wtb_k3_msf_seed42/convnext_wds_wtb_k3_msf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9721 |
| Accuracy | 0.9722 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9722 |
| Balanced Acc | 0.9716 |
| Macro-Sensitivity | 0.9716 |
| Macro-Specificity | 0.9859 |
| Macro-Precision | 0.9728 |
| AUC-PR (Macro) | 0.9927 |
| Cohen's Kappa | 0.9582 |
| MCC | 0.9583 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.53G |
| FLOPs | 9.05G |
| Params | 28.72M |
| Pure-Inference FPS | 1157.6 (0.86 ms/张) |
| End-to-End FPS | 231.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9755 | 0.9573 | 0.9642 | 0.9836 | 0.9949 | 0.9913 | 208 |
| Malignant_lesions | 0.9489 | 0.9900 | 0.9766 | 0.9625 | 0.9778 | 0.9945 | 0.9876 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9922 | 0.9845 | 0.9896 | 0.9974 | 0.9996 | 0.9993 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=9 TN=359 FN=6
- **Malignant_lesions**: TP=167 FP=4 TN=396 FN=9
- **body_of_stomach_normal**: TP=191 FP=3 TN=381 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 8 | 167 | 1 |
| body_of_stomach_normal | 1 | 0 | 191 |

## 核心指标 95% Bootstrap CI

| 指标 | 95% CI |
|---|---|
| macro_f1 | [0.9576, 0.9855] |
| accuracy | [0.9583, 0.9861] |
| auc_roc_macro | [0.9932, 0.9990] |
| macro_sensitivity | [0.9571, 0.9852] |
| macro_specificity | [0.9787, 0.9928] |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9573    0.9712    0.9642       208
     Malignant_lesions     0.9766    0.9489    0.9625       176
body_of_stomach_normal     0.9845    0.9948    0.9896       192

              accuracy                         0.9722       576
             macro avg     0.9728    0.9716    0.9721       576
          weighted avg     0.9723    0.9722    0.9722       576

```