# fast_t0_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_dce4m_seed42/fast_t0_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9335 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9934 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9338 |
| Balanced Acc | 0.9314 |
| Macro-Sensitivity | 0.9314 |
| Macro-Specificity | 0.9663 |
| Macro-Precision | 0.9381 |
| AUC-PR (Macro) | 0.9871 |
| Cohen's Kappa | 0.9006 |
| MCC | 0.9018 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 6754.4 (0.15 ms/张) |
| End-to-End FPS | 8.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9321 | 0.8884 | 0.9213 | 0.9744 | 0.9898 | 0.9835 | 208 |
| Malignant_lesions | 0.8636 | 0.9850 | 0.9620 | 0.9102 | 0.9426 | 0.9913 | 0.9793 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9818 | 0.9639 | 0.9689 | 0.9869 | 0.9992 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=25 TN=343 FN=9
- **Malignant_lesions**: TP=152 FP=6 TN=394 FN=24
- **body_of_stomach_normal**: TP=187 FP=7 TN=377 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 6 | 3 |
| Malignant_lesions | 20 | 152 | 4 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8884    0.9567    0.9213       208
     Malignant_lesions     0.9620    0.8636    0.9102       176
body_of_stomach_normal     0.9639    0.9740    0.9689       192

              accuracy                         0.9340       576
             macro avg     0.9381    0.9314    0.9335       576
          weighted avg     0.9361    0.9340    0.9338       576

```