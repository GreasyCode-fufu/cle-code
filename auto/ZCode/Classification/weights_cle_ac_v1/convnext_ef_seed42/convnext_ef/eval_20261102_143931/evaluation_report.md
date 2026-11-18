# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_ef_seed42/convnext_ef/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9490 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9923 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9494 |
| Balanced Acc | 0.9476 |
| Macro-Sensitivity | 0.9476 |
| Macro-Specificity | 0.9743 |
| Macro-Precision | 0.9519 |
| AUC-PR (Macro) | 0.9856 |
| Cohen's Kappa | 0.9242 |
| MCC | 0.9249 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 642.0 (1.56 ms/张) |
| End-to-End FPS | 214.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9484 | 0.9132 | 0.9368 | 0.9776 | 0.9904 | 0.9842 | 208 |
| Malignant_lesions | 0.8864 | 0.9850 | 0.9630 | 0.9231 | 0.9517 | 0.9865 | 0.9728 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9896 | 0.9795 | 0.9871 | 0.9974 | 0.9999 | 0.9997 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=19 TN=349 FN=8
- **Malignant_lesions**: TP=156 FP=6 TN=394 FN=20
- **body_of_stomach_normal**: TP=191 FP=4 TN=380 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 18 | 156 | 2 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9132    0.9615    0.9368       208
     Malignant_lesions     0.9630    0.8864    0.9231       176
body_of_stomach_normal     0.9795    0.9948    0.9871       192

              accuracy                         0.9497       576
             macro avg     0.9519    0.9476    0.9490       576
          weighted avg     0.9505    0.9497    0.9494       576

```