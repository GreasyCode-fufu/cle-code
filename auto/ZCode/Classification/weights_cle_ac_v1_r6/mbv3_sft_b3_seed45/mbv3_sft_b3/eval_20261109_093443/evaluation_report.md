# mbv3_sft_b3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b3_seed45/mbv3_sft_b3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9367 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9932 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9371 |
| Balanced Acc | 0.9352 |
| Macro-Sensitivity | 0.9352 |
| Macro-Specificity | 0.9682 |
| Macro-Precision | 0.9398 |
| AUC-PR (Macro) | 0.9871 |
| Cohen's Kappa | 0.9058 |
| MCC | 0.9066 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4337.4 (0.23 ms/张) |
| End-to-End FPS | 216.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9402 | 0.9000 | 0.9252 | 0.9719 | 0.9891 | 0.9817 | 208 |
| Malignant_lesions | 0.8693 | 0.9800 | 0.9503 | 0.9080 | 0.9446 | 0.9921 | 0.9828 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9844 | 0.9692 | 0.9767 | 0.9921 | 0.9982 | 0.9968 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=22 TN=346 FN=10
- **Malignant_lesions**: TP=153 FP=8 TN=392 FN=23
- **body_of_stomach_normal**: TP=189 FP=6 TN=378 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 8 | 2 |
| Malignant_lesions | 19 | 153 | 4 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9000    0.9519    0.9252       208
     Malignant_lesions     0.9503    0.8693    0.9080       176
body_of_stomach_normal     0.9692    0.9844    0.9767       192

              accuracy                         0.9375       576
             macro avg     0.9398    0.9352    0.9367       576
          weighted avg     0.9384    0.9375    0.9371       576

```