# mbv3_sft_b1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b1_seed45/mbv3_sft_b1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9533 |
| Accuracy | 0.9625 |
| Macro-AUROC (OvR) | 0.9957 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9626 |
| Balanced Acc | 0.9546 |
| Macro-Sensitivity | 0.9546 |
| Macro-Specificity | 0.9813 |
| Macro-Precision | 0.9520 |
| AUC-PR (Macro) | 0.9858 |
| Cohen's Kappa | 0.9405 |
| MCC | 0.9405 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2278.3 (0.44 ms/张) |
| End-to-End FPS | 120.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9590 | 0.9741 | 0.9689 | 0.9639 | 0.9658 | 0.9956 | 0.9953 | 195 |
| Malignant_lesions | 0.9114 | 0.9770 | 0.9000 | 0.9057 | 0.9798 | 0.9917 | 0.9626 | 79 |
| body_of_stomach_normal | 0.9935 | 0.9927 | 0.9870 | 0.9902 | 0.9963 | 0.9998 | 0.9996 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=6 TN=226 FN=8
- **Malignant_lesions**: TP=72 FP=8 TN=340 FN=7
- **body_of_stomach_normal**: TP=152 FP=2 TN=272 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 8 | 0 |
| Malignant_lesions | 5 | 72 | 2 |
| body_of_stomach_normal | 1 | 0 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9689    0.9590    0.9639       195
     Malignant_lesions     0.9000    0.9114    0.9057        79
body_of_stomach_normal     0.9870    0.9935    0.9902       153

              accuracy                         0.9625       427
             macro avg     0.9520    0.9546    0.9533       427
          weighted avg     0.9626    0.9625    0.9626       427

```