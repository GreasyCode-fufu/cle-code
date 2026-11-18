# convnext_egb_scg 评估报告

- 权重: `/home/login/data/cle/auto/Network/CLE-Net/historical_protocol_codex_20261109/runs/convnext_egb_scg_seed43/convnext_egb_scg/model_best.pth`
- 测试集: `/home/login/data/cle/auto/Network/CLE-Net/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9585 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9960 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9585 |
| Balanced Acc | 0.9585 |
| Macro-Sensitivity | 0.9585 |
| Macro-Specificity | 0.9791 |
| Macro-Precision | 0.9586 |
| AUC-PR (Macro) | 0.9924 |
| Cohen's Kappa | 0.9374 |
| MCC | 0.9374 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 1001.0 (1.00 ms/张) |
| End-to-End FPS | 248.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9674 | 0.9426 | 0.9448 | 0.9700 | 0.9941 | 0.9902 | 208 |
| Malignant_lesions | 0.9545 | 0.9725 | 0.9385 | 0.9465 | 0.9798 | 0.9946 | 0.9883 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9974 | 0.9947 | 0.9842 | 0.9871 | 0.9992 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=12 TN=356 FN=11
- **Malignant_lesions**: TP=168 FP=11 TN=389 FN=8
- **body_of_stomach_normal**: TP=187 FP=1 TN=383 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 8 | 168 | 0 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9426    0.9471    0.9448       208
     Malignant_lesions     0.9385    0.9545    0.9465       176
body_of_stomach_normal     0.9947    0.9740    0.9842       192

              accuracy                         0.9583       576
             macro avg     0.9586    0.9585    0.9585       576
          weighted avg     0.9587    0.9583    0.9585       576

```