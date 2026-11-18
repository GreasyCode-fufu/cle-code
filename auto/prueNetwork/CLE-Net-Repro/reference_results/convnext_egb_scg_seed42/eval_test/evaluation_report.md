# convnext_egb_scg 评估报告

- 权重: `/home/login/data/cle/auto/Network/CLE-Net/historical_protocol_codex_20261109/runs/convnext_egb_scg_seed42/convnext_egb_scg/model_best.pth`
- 测试集: `/home/login/data/cle/auto/Network/CLE-Net/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9617 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9962 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9618 |
| Balanced Acc | 0.9617 |
| Macro-Sensitivity | 0.9617 |
| Macro-Specificity | 0.9809 |
| Macro-Precision | 0.9616 |
| AUC-PR (Macro) | 0.9928 |
| Cohen's Kappa | 0.9426 |
| MCC | 0.9426 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 1002.6 (1.00 ms/张) |
| End-to-End FPS | 249.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9755 | 0.9567 | 0.9567 | 0.9755 | 0.9949 | 0.9919 | 208 |
| Malignant_lesions | 0.9545 | 0.9775 | 0.9492 | 0.9518 | 0.9799 | 0.9949 | 0.9886 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9896 | 0.9791 | 0.9765 | 0.9870 | 0.9990 | 0.9980 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=9 TN=359 FN=9
- **Malignant_lesions**: TP=168 FP=9 TN=391 FN=8
- **body_of_stomach_normal**: TP=187 FP=4 TN=380 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 5 | 168 | 3 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9567    0.9567    0.9567       208
     Malignant_lesions     0.9492    0.9545    0.9518       176
body_of_stomach_normal     0.9791    0.9740    0.9765       192

              accuracy                         0.9618       576
             macro avg     0.9616    0.9617    0.9617       576
          weighted avg     0.9619    0.9618    0.9618       576

```