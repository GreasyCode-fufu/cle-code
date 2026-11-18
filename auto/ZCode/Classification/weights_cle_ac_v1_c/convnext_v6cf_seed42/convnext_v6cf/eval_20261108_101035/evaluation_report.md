# convnext_v6cf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v6cf_seed42/convnext_v6cf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9458 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9951 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9464 |
| Balanced Acc | 0.9459 |
| Macro-Sensitivity | 0.9459 |
| Macro-Specificity | 0.9732 |
| Macro-Precision | 0.9458 |
| AUC-PR (Macro) | 0.9904 |
| Cohen's Kappa | 0.9191 |
| MCC | 0.9192 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 5.79G |
| FLOPs | 11.58G |
| Params | 30.61M |
| Pure-Inference FPS | 403.5 (2.48 ms/张) |
| End-to-End FPS | 171.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9674 | 0.9423 | 0.9423 | 0.9674 | 0.9937 | 0.9902 | 208 |
| Malignant_lesions | 0.9318 | 0.9600 | 0.9111 | 0.9213 | 0.9697 | 0.9922 | 0.9822 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9922 | 0.9840 | 0.9737 | 0.9820 | 0.9994 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=12 TN=356 FN=12
- **Malignant_lesions**: TP=164 FP=16 TN=384 FN=12
- **body_of_stomach_normal**: TP=185 FP=3 TN=381 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 12 | 0 |
| Malignant_lesions | 9 | 164 | 3 |
| body_of_stomach_normal | 3 | 4 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9423    0.9423    0.9423       208
     Malignant_lesions     0.9111    0.9318    0.9213       176
body_of_stomach_normal     0.9840    0.9635    0.9737       192

              accuracy                         0.9462       576
             macro avg     0.9458    0.9459    0.9458       576
          weighted avg     0.9467    0.9462    0.9464       576

```