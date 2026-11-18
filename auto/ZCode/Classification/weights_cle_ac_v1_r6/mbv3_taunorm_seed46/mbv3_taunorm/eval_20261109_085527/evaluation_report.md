# mbv3_taunorm 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_taunorm_seed46/mbv3_taunorm/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9465 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9890 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9464 |
| Balanced Acc | 0.9459 |
| Macro-Sensitivity | 0.9459 |
| Macro-Specificity | 0.9728 |
| Macro-Precision | 0.9474 |
| AUC-PR (Macro) | 0.9766 |
| Cohen's Kappa | 0.9190 |
| MCC | 0.9192 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 2809.8 (0.36 ms/张) |
| End-to-End FPS | 8.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9511 | 0.9159 | 0.9289 | 0.9669 | 0.9835 | 0.9681 | 208 |
| Malignant_lesions | 0.9318 | 0.9700 | 0.9318 | 0.9318 | 0.9700 | 0.9858 | 0.9654 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9974 | 0.9946 | 0.9788 | 0.9821 | 0.9976 | 0.9962 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=18 TN=350 FN=12
- **Malignant_lesions**: TP=164 FP=12 TN=388 FN=12
- **body_of_stomach_normal**: TP=185 FP=1 TN=383 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 12 | 0 |
| Malignant_lesions | 11 | 164 | 1 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9159    0.9423    0.9289       208
     Malignant_lesions     0.9318    0.9318    0.9318       176
body_of_stomach_normal     0.9946    0.9635    0.9788       192

              accuracy                         0.9462       576
             macro avg     0.9474    0.9459    0.9465       576
          weighted avg     0.9470    0.9462    0.9464       576

```