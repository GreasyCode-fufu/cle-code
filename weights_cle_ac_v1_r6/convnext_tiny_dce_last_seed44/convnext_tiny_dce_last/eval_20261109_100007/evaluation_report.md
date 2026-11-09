# convnext_tiny_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_dce_last_seed44/convnext_tiny_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9639 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9972 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9718 |
| Balanced Acc | 0.9619 |
| Macro-Sensitivity | 0.9619 |
| Macro-Specificity | 0.9854 |
| Macro-Precision | 0.9660 |
| AUC-PR (Macro) | 0.9906 |
| Cohen's Kappa | 0.9552 |
| MCC | 0.9553 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1663.0 (0.60 ms/张) |
| End-to-End FPS | 227.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9744 | 0.9741 | 0.9694 | 0.9719 | 0.9784 | 0.9971 | 0.9968 | 195 |
| Malignant_lesions | 0.9114 | 0.9856 | 0.9351 | 0.9231 | 0.9800 | 0.9946 | 0.9749 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=6 TN=226 FN=5
- **Malignant_lesions**: TP=72 FP=5 TN=343 FN=7
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 5 | 0 |
| Malignant_lesions | 6 | 72 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9694    0.9744    0.9719       195
     Malignant_lesions     0.9351    0.9114    0.9231        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9719       427
             macro avg     0.9660    0.9619    0.9639       427
          weighted avg     0.9717    0.9719    0.9718       427

```