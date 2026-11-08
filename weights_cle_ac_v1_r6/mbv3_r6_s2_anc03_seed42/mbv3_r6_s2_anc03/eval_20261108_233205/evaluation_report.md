# mbv3_r6_s2_anc03 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_anc03_seed42/mbv3_r6_s2_anc03/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9458 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9937 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9459 |
| Balanced Acc | 0.9458 |
| Macro-Sensitivity | 0.9458 |
| Macro-Specificity | 0.9729 |
| Macro-Precision | 0.9461 |
| AUC-PR (Macro) | 0.9881 |
| Cohen's Kappa | 0.9191 |
| MCC | 0.9192 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 2976.7 (0.34 ms/张) |
| End-to-End FPS | 106.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9647 | 0.9369 | 0.9324 | 0.9595 | 0.9900 | 0.9838 | 208 |
| Malignant_lesions | 0.9148 | 0.9750 | 0.9415 | 0.9280 | 0.9630 | 0.9923 | 0.9831 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9792 | 0.9598 | 0.9770 | 0.9973 | 0.9987 | 0.9973 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=13 TN=355 FN=15
- **Malignant_lesions**: TP=161 FP=10 TN=390 FN=15
- **body_of_stomach_normal**: TP=191 FP=8 TN=376 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 10 | 5 |
| Malignant_lesions | 12 | 161 | 3 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9369    0.9279    0.9324       208
     Malignant_lesions     0.9415    0.9148    0.9280       176
body_of_stomach_normal     0.9598    0.9948    0.9770       192

              accuracy                         0.9462       576
             macro avg     0.9461    0.9458    0.9458       576
          weighted avg     0.9459    0.9462    0.9459       576

```