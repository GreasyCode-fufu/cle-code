# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_egb_scg_seed46/convnext_egb_scg/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9582 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9940 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9584 |
| Balanced Acc | 0.9571 |
| Macro-Sensitivity | 0.9571 |
| Macro-Specificity | 0.9788 |
| Macro-Precision | 0.9599 |
| AUC-PR (Macro) | 0.9893 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9376 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 294.9 (3.39 ms/张) |
| End-to-End FPS | 145.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9565 | 0.9263 | 0.9459 | 0.9805 | 0.9926 | 0.9870 | 208 |
| Malignant_lesions | 0.9205 | 0.9825 | 0.9586 | 0.9391 | 0.9656 | 0.9895 | 0.9814 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9974 | 0.9947 | 0.9895 | 0.9922 | 0.9998 | 0.9996 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=16 TN=352 FN=7
- **Malignant_lesions**: TP=162 FP=7 TN=393 FN=14
- **body_of_stomach_normal**: TP=189 FP=1 TN=383 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 14 | 162 | 0 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9263    0.9663    0.9459       208
     Malignant_lesions     0.9586    0.9205    0.9391       176
body_of_stomach_normal     0.9947    0.9844    0.9895       192

              accuracy                         0.9583       576
             macro avg     0.9599    0.9571    0.9582       576
          weighted avg     0.9590    0.9583    0.9584       576

```