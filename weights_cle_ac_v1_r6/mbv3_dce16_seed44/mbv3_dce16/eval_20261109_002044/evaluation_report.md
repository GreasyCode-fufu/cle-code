# mbv3_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce16_seed44/mbv3_dce16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9515 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9923 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9515 |
| Balanced Acc | 0.9517 |
| Macro-Sensitivity | 0.9517 |
| Macro-Specificity | 0.9757 |
| Macro-Precision | 0.9515 |
| AUC-PR (Macro) | 0.9854 |
| Cohen's Kappa | 0.9269 |
| MCC | 0.9270 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 5233.7 (0.19 ms/张) |
| End-to-End FPS | 22.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9674 | 0.9423 | 0.9423 | 0.9674 | 0.9870 | 0.9794 | 208 |
| Malignant_lesions | 0.9545 | 0.9675 | 0.9282 | 0.9412 | 0.9797 | 0.9910 | 0.9791 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9922 | 0.9840 | 0.9710 | 0.9794 | 0.9988 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=12 TN=356 FN=12
- **Malignant_lesions**: TP=168 FP=13 TN=387 FN=8
- **body_of_stomach_normal**: TP=184 FP=3 TN=381 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 10 | 2 |
| Malignant_lesions | 7 | 168 | 1 |
| body_of_stomach_normal | 5 | 3 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9423    0.9423    0.9423       208
     Malignant_lesions     0.9282    0.9545    0.9412       176
body_of_stomach_normal     0.9840    0.9583    0.9710       192

              accuracy                         0.9514       576
             macro avg     0.9515    0.9517    0.9515       576
          weighted avg     0.9519    0.9514    0.9515       576

```