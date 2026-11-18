# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz8_dce4m_seed42/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9680 |
| Accuracy | 0.9742 |
| Macro-AUROC (OvR) | 0.9976 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9745 |
| Balanced Acc | 0.9737 |
| Macro-Sensitivity | 0.9737 |
| Macro-Specificity | 0.9880 |
| Macro-Precision | 0.9630 |
| AUC-PR (Macro) | 0.9923 |
| Cohen's Kappa | 0.9592 |
| MCC | 0.9595 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3262.0 (0.31 ms/张) |
| End-to-End FPS | 16.9 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9590 | 0.9871 | 0.9842 | 0.9714 | 0.9662 | 0.9975 | 0.9971 | 195 |
| Malignant_lesions | 0.9620 | 0.9770 | 0.9048 | 0.9325 | 0.9913 | 0.9953 | 0.9796 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=3 TN=229 FN=8
- **Malignant_lesions**: TP=76 FP=8 TN=340 FN=3
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 8 | 0 |
| Malignant_lesions | 3 | 76 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9842    0.9590    0.9714       195
     Malignant_lesions     0.9048    0.9620    0.9325        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9742       427
             macro avg     0.9630    0.9737    0.9680       427
          weighted avg     0.9752    0.9742    0.9745       427

```