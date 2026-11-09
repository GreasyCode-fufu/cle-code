# mbv3_sft_b2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b2_seed44/mbv3_sft_b2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9600 |
| Accuracy | 0.9672 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9674 |
| Balanced Acc | 0.9630 |
| Macro-Sensitivity | 0.9630 |
| Macro-Specificity | 0.9837 |
| Macro-Precision | 0.9571 |
| AUC-PR (Macro) | 0.9881 |
| Cohen's Kappa | 0.9480 |
| MCC | 0.9481 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2631.8 (0.38 ms/张) |
| End-to-End FPS | 173.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9590 | 0.9741 | 0.9689 | 0.9639 | 0.9658 | 0.9957 | 0.9949 | 195 |
| Malignant_lesions | 0.9367 | 0.9770 | 0.9024 | 0.9193 | 0.9855 | 0.9926 | 0.9694 | 79 |
| body_of_stomach_normal | 0.9935 | 1.0000 | 1.0000 | 0.9967 | 0.9964 | 1.0000 | 0.9999 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=6 TN=226 FN=8
- **Malignant_lesions**: TP=74 FP=8 TN=340 FN=5
- **body_of_stomach_normal**: TP=152 FP=0 TN=274 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 8 | 0 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 1 | 0 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9689    0.9590    0.9639       195
     Malignant_lesions     0.9024    0.9367    0.9193        79
body_of_stomach_normal     1.0000    0.9935    0.9967       153

              accuracy                         0.9672       427
             macro avg     0.9571    0.9630    0.9600       427
          weighted avg     0.9678    0.9672    0.9674       427

```