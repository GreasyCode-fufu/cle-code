# fast_t0_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_seed43/fast_t0_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9389 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9908 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9391 |
| Balanced Acc | 0.9375 |
| Macro-Sensitivity | 0.9375 |
| Macro-Specificity | 0.9691 |
| Macro-Precision | 0.9413 |
| AUC-PR (Macro) | 0.9802 |
| Cohen's Kappa | 0.9085 |
| MCC | 0.9090 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 5124.1 (0.20 ms/张) |
| End-to-End FPS | 122.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9429 | 0.9037 | 0.9249 | 0.9693 | 0.9858 | 0.9780 | 208 |
| Malignant_lesions | 0.8864 | 0.9800 | 0.9512 | 0.9176 | 0.9515 | 0.9873 | 0.9638 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9844 | 0.9691 | 0.9741 | 0.9895 | 0.9994 | 0.9989 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=21 TN=347 FN=11
- **Malignant_lesions**: TP=156 FP=8 TN=392 FN=20
- **body_of_stomach_normal**: TP=188 FP=6 TN=378 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 8 | 3 |
| Malignant_lesions | 17 | 156 | 3 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9037    0.9471    0.9249       208
     Malignant_lesions     0.9512    0.8864    0.9176       176
body_of_stomach_normal     0.9691    0.9792    0.9741       192

              accuracy                         0.9392       576
             macro avg     0.9413    0.9375    0.9389       576
          weighted avg     0.9400    0.9392    0.9391       576

```