# mbv3_sft_b4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b4_seed42/mbv3_sft_b4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9639 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9717 |
| Balanced Acc | 0.9594 |
| Macro-Sensitivity | 0.9594 |
| Macro-Specificity | 0.9847 |
| Macro-Precision | 0.9688 |
| AUC-PR (Macro) | 0.9898 |
| Cohen's Kappa | 0.9551 |
| MCC | 0.9553 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1793.7 (0.56 ms/张) |
| End-to-End FPS | 32.7 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9655 | 0.9598 | 0.9695 | 0.9825 | 0.9960 | 0.9952 | 195 |
| Malignant_lesions | 0.8987 | 0.9885 | 0.9467 | 0.9221 | 0.9773 | 0.9931 | 0.9741 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=8 TN=224 FN=4
- **Malignant_lesions**: TP=71 FP=4 TN=344 FN=8
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 4 | 0 |
| Malignant_lesions | 8 | 71 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9598    0.9795    0.9695       195
     Malignant_lesions     0.9467    0.8987    0.9221        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9719       427
             macro avg     0.9688    0.9594    0.9639       427
          weighted avg     0.9718    0.9719    0.9717       427

```