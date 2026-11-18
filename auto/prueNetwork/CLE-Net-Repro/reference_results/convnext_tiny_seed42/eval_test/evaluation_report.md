# convnext_tiny 评估报告

- 权重: `/home/login/data/cle/auto/Network/CLE-Net/historical_protocol_codex_20261109/runs/convnext_tiny_seed42/convnext_tiny/model_best.pth`
- 测试集: `/home/login/data/cle/auto/Network/CLE-Net/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9636 |
| Accuracy | 0.9635 |
| Macro-AUROC (OvR) | 0.9965 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9637 |
| Balanced Acc | 0.9636 |
| Macro-Sensitivity | 0.9636 |
| Macro-Specificity | 0.9817 |
| Macro-Precision | 0.9637 |
| AUC-PR (Macro) | 0.9935 |
| Cohen's Kappa | 0.9452 |
| MCC | 0.9452 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1717.9 (0.58 ms/张) |
| End-to-End FPS | 274.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9728 | 0.9522 | 0.9544 | 0.9755 | 0.9952 | 0.9921 | 208 |
| Malignant_lesions | 0.9602 | 0.9750 | 0.9441 | 0.9521 | 0.9824 | 0.9948 | 0.9892 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9974 | 0.9947 | 0.9842 | 0.9871 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=10 TN=358 FN=9
- **Malignant_lesions**: TP=169 FP=10 TN=390 FN=7
- **body_of_stomach_normal**: TP=187 FP=1 TN=383 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 7 | 169 | 0 |
| body_of_stomach_normal | 3 | 2 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9522    0.9567    0.9544       208
     Malignant_lesions     0.9441    0.9602    0.9521       176
body_of_stomach_normal     0.9947    0.9740    0.9842       192

              accuracy                         0.9635       576
             macro avg     0.9637    0.9636    0.9636       576
          weighted avg     0.9639    0.9635    0.9637       576

```