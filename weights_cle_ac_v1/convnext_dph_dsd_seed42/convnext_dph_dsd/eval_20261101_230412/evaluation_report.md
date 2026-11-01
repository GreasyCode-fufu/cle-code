# convnext_dph_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_dsd_seed42/convnext_dph_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9853 |
| Accuracy | 0.9883 |
| Macro-AUROC (OvR) | 0.9989 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9884 |
| Balanced Acc | 0.9889 |
| Macro-Sensitivity | 0.9889 |
| Macro-Specificity | 0.9947 |
| Macro-Precision | 0.9820 |
| AUC-PR (Macro) | 0.9964 |
| Cohen's Kappa | 0.9814 |
| MCC | 0.9815 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 28.68M |
| Pure-Inference FPS | 1004.9 (1.00 ms/张) |
| End-to-End FPS | 231.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9957 | 0.9948 | 0.9871 | 0.9830 | 0.9988 | 0.9987 | 195 |
| Malignant_lesions | 0.9873 | 0.9885 | 0.9512 | 0.9689 | 0.9971 | 0.9980 | 0.9906 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=1 TN=231 FN=4
- **Malignant_lesions**: TP=78 FP=4 TN=344 FN=1
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 4 | 0 |
| Malignant_lesions | 1 | 78 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9948    0.9795    0.9871       195
     Malignant_lesions     0.9512    0.9873    0.9689        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9883       427
             macro avg     0.9820    0.9889    0.9853       427
          weighted avg     0.9886    0.9883    0.9884       427

```