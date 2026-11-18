# fast_t0_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_dce_last_seed43/fast_t0_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9405 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9921 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9407 |
| Balanced Acc | 0.9393 |
| Macro-Sensitivity | 0.9393 |
| Macro-Specificity | 0.9700 |
| Macro-Precision | 0.9428 |
| AUC-PR (Macro) | 0.9843 |
| Cohen's Kappa | 0.9111 |
| MCC | 0.9116 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 7897.6 (0.13 ms/张) |
| End-to-End FPS | 8.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9457 | 0.9078 | 0.9271 | 0.9694 | 0.9875 | 0.9810 | 208 |
| Malignant_lesions | 0.8864 | 0.9800 | 0.9512 | 0.9176 | 0.9515 | 0.9896 | 0.9734 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9844 | 0.9692 | 0.9767 | 0.9921 | 0.9991 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=20 TN=348 FN=11
- **Malignant_lesions**: TP=156 FP=8 TN=392 FN=20
- **body_of_stomach_normal**: TP=189 FP=6 TN=378 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 8 | 3 |
| Malignant_lesions | 17 | 156 | 3 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9078    0.9471    0.9271       208
     Malignant_lesions     0.9512    0.8864    0.9176       176
body_of_stomach_normal     0.9692    0.9844    0.9767       192

              accuracy                         0.9410       576
             macro avg     0.9428    0.9393    0.9405       576
          weighted avg     0.9416    0.9410    0.9407       576

```