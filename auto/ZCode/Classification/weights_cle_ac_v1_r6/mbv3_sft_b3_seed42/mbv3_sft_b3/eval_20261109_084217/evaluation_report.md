# mbv3_sft_b3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b3_seed42/mbv3_sft_b3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9117 |
| Accuracy | 0.9132 |
| Macro-AUROC (OvR) | 0.9887 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9125 |
| Balanced Acc | 0.9093 |
| Macro-Sensitivity | 0.9093 |
| Macro-Specificity | 0.9557 |
| Macro-Precision | 0.9188 |
| AUC-PR (Macro) | 0.9783 |
| Cohen's Kappa | 0.8690 |
| MCC | 0.8712 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3489.3 (0.29 ms/张) |
| End-to-End FPS | 27.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9103 | 0.8571 | 0.9021 | 0.9710 | 0.9849 | 0.9754 | 208 |
| Malignant_lesions | 0.8125 | 0.9775 | 0.9408 | 0.8720 | 0.9222 | 0.9845 | 0.9654 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9792 | 0.9585 | 0.9610 | 0.9817 | 0.9968 | 0.9943 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=33 TN=335 FN=10
- **Malignant_lesions**: TP=143 FP=9 TN=391 FN=33
- **body_of_stomach_normal**: TP=185 FP=8 TN=376 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 9 | 1 |
| Malignant_lesions | 26 | 143 | 7 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8571    0.9519    0.9021       208
     Malignant_lesions     0.9408    0.8125    0.8720       176
body_of_stomach_normal     0.9585    0.9635    0.9610       192

              accuracy                         0.9132       576
             macro avg     0.9188    0.9093    0.9117       576
          weighted avg     0.9165    0.9132    0.9125       576

```