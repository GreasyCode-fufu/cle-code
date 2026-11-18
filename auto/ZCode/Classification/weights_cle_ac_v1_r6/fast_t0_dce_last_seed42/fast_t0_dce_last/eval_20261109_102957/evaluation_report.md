# fast_t0_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_dce_last_seed42/fast_t0_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9369 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9923 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9373 |
| Balanced Acc | 0.9345 |
| Macro-Sensitivity | 0.9345 |
| Macro-Specificity | 0.9679 |
| Macro-Precision | 0.9429 |
| AUC-PR (Macro) | 0.9850 |
| Cohen's Kappa | 0.9058 |
| MCC | 0.9074 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 7419.9 (0.13 ms/张) |
| End-to-End FPS | 8.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9266 | 0.8816 | 0.9220 | 0.9799 | 0.9881 | 0.9820 | 208 |
| Malignant_lesions | 0.8580 | 0.9875 | 0.9679 | 0.9096 | 0.9405 | 0.9900 | 0.9751 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9896 | 0.9792 | 0.9792 | 0.9896 | 0.9988 | 0.9979 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=27 TN=341 FN=7
- **Malignant_lesions**: TP=151 FP=5 TN=395 FN=25
- **body_of_stomach_normal**: TP=188 FP=4 TN=380 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 23 | 151 | 2 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8816    0.9663    0.9220       208
     Malignant_lesions     0.9679    0.8580    0.9096       176
body_of_stomach_normal     0.9792    0.9792    0.9792       192

              accuracy                         0.9375       576
             macro avg     0.9429    0.9345    0.9369       576
          weighted avg     0.9405    0.9375    0.9373       576

```