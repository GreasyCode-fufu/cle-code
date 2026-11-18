# convnext_dph_mfm2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_mfm2_seed43/convnext_dph_mfm2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9511 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9945 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9517 |
| Balanced Acc | 0.9517 |
| Macro-Sensitivity | 0.9517 |
| Macro-Specificity | 0.9760 |
| Macro-Precision | 0.9511 |
| AUC-PR (Macro) | 0.9896 |
| Cohen's Kappa | 0.9270 |
| MCC | 0.9273 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 678.3 (1.47 ms/张) |
| End-to-End FPS | 207.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9783 | 0.9608 | 0.9515 | 0.9677 | 0.9928 | 0.9884 | 208 |
| Malignant_lesions | 0.9545 | 0.9550 | 0.9032 | 0.9282 | 0.9795 | 0.9914 | 0.9822 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9948 | 0.9892 | 0.9735 | 0.9795 | 0.9992 | 0.9983 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=8 TN=360 FN=12
- **Malignant_lesions**: TP=168 FP=18 TN=382 FN=8
- **body_of_stomach_normal**: TP=184 FP=2 TN=382 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 11 | 1 |
| Malignant_lesions | 7 | 168 | 1 |
| body_of_stomach_normal | 1 | 7 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9608    0.9423    0.9515       208
     Malignant_lesions     0.9032    0.9545    0.9282       176
body_of_stomach_normal     0.9892    0.9583    0.9735       192

              accuracy                         0.9514       576
             macro avg     0.9511    0.9517    0.9511       576
          weighted avg     0.9527    0.9514    0.9517       576

```