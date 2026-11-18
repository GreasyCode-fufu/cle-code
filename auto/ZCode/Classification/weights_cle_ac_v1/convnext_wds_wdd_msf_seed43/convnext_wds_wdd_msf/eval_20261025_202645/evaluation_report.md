# convnext_wds_wdd_msf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wdd_msf_seed43/convnext_wds_wdd_msf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9600 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9960 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9601 |
| Balanced Acc | 0.9606 |
| Macro-Sensitivity | 0.9606 |
| Macro-Specificity | 0.9801 |
| Macro-Precision | 0.9596 |
| AUC-PR (Macro) | 0.9923 |
| Cohen's Kappa | 0.9400 |
| MCC | 0.9401 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.68G |
| FLOPs | 9.36G |
| Params | 29.80M |
| Pure-Inference FPS | 497.3 (2.01 ms/张) |
| End-to-End FPS | 191.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9783 | 0.9608 | 0.9515 | 0.9677 | 0.9940 | 0.9906 | 208 |
| Malignant_lesions | 0.9602 | 0.9725 | 0.9389 | 0.9494 | 0.9823 | 0.9949 | 0.9881 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9896 | 0.9792 | 0.9792 | 0.9896 | 0.9990 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=8 TN=360 FN=12
- **Malignant_lesions**: TP=169 FP=11 TN=389 FN=7
- **body_of_stomach_normal**: TP=188 FP=4 TN=380 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 10 | 2 |
| Malignant_lesions | 5 | 169 | 2 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9608    0.9423    0.9515       208
     Malignant_lesions     0.9389    0.9602    0.9494       176
body_of_stomach_normal     0.9792    0.9792    0.9792       192

              accuracy                         0.9601       576
             macro avg     0.9596    0.9606    0.9600       576
          weighted avg     0.9602    0.9601    0.9601       576

```