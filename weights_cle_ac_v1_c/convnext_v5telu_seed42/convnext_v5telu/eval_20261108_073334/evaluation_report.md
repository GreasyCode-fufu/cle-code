# convnext_v5telu 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5telu_seed42/convnext_v5telu/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9532 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9953 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9532 |
| Balanced Acc | 0.9514 |
| Macro-Sensitivity | 0.9514 |
| Macro-Specificity | 0.9760 |
| Macro-Precision | 0.9563 |
| AUC-PR (Macro) | 0.9909 |
| Cohen's Kappa | 0.9294 |
| MCC | 0.9301 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 408.1 (2.45 ms/张) |
| End-to-End FPS | 176.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9457 | 0.9099 | 0.9395 | 0.9831 | 0.9933 | 0.9893 | 208 |
| Malignant_lesions | 0.9091 | 0.9875 | 0.9697 | 0.9384 | 0.9611 | 0.9933 | 0.9847 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9948 | 0.9894 | 0.9816 | 0.9871 | 0.9993 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=20 TN=348 FN=6
- **Malignant_lesions**: TP=160 FP=5 TN=395 FN=16
- **body_of_stomach_normal**: TP=187 FP=2 TN=382 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 15 | 160 | 1 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9099    0.9712    0.9395       208
     Malignant_lesions     0.9697    0.9091    0.9384       176
body_of_stomach_normal     0.9894    0.9740    0.9816       192

              accuracy                         0.9531       576
             macro avg     0.9563    0.9514    0.9532       576
          weighted avg     0.9547    0.9531    0.9532       576

```