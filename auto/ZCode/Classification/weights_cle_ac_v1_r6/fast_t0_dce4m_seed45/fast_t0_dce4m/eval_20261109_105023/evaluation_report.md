# fast_t0_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_dce4m_seed45/fast_t0_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9277 |
| Accuracy | 0.9288 |
| Macro-AUROC (OvR) | 0.9923 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9283 |
| Balanced Acc | 0.9250 |
| Macro-Sensitivity | 0.9250 |
| Macro-Specificity | 0.9634 |
| Macro-Precision | 0.9354 |
| AUC-PR (Macro) | 0.9853 |
| Cohen's Kappa | 0.8926 |
| MCC | 0.8950 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 7418.6 (0.13 ms/张) |
| End-to-End FPS | 8.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9158 | 0.8664 | 0.9136 | 0.9797 | 0.9885 | 0.9799 | 208 |
| Malignant_lesions | 0.8295 | 0.9850 | 0.9605 | 0.8902 | 0.9292 | 0.9898 | 0.9784 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9896 | 0.9792 | 0.9792 | 0.9896 | 0.9987 | 0.9975 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=31 TN=337 FN=7
- **Malignant_lesions**: TP=146 FP=6 TN=394 FN=30
- **body_of_stomach_normal**: TP=188 FP=4 TN=380 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 27 | 146 | 3 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8664    0.9663    0.9136       208
     Malignant_lesions     0.9605    0.8295    0.8902       176
body_of_stomach_normal     0.9792    0.9792    0.9792       192

              accuracy                         0.9288       576
             macro avg     0.9354    0.9250    0.9277       576
          weighted avg     0.9327    0.9288    0.9283       576

```