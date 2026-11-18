# convnext_wds_wtb_k3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wtb_k3_seed42/convnext_wds_wtb_k3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9621 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9960 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9619 |
| Balanced Acc | 0.9613 |
| Macro-Sensitivity | 0.9613 |
| Macro-Specificity | 0.9806 |
| Macro-Precision | 0.9633 |
| AUC-PR (Macro) | 0.9928 |
| Cohen's Kappa | 0.9425 |
| MCC | 0.9428 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.50G |
| FLOPs | 9.01G |
| Params | 28.28M |
| Pure-Inference FPS | 1448.2 (0.69 ms/张) |
| End-to-End FPS | 263.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9620 | 0.9352 | 0.9528 | 0.9833 | 0.9943 | 0.9902 | 208 |
| Malignant_lesions | 0.9545 | 0.9850 | 0.9655 | 0.9600 | 0.9801 | 0.9945 | 0.9899 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9948 | 0.9892 | 0.9735 | 0.9795 | 0.9991 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=14 TN=354 FN=6
- **Malignant_lesions**: TP=168 FP=6 TN=394 FN=8
- **body_of_stomach_normal**: TP=184 FP=2 TN=382 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 7 | 168 | 1 |
| body_of_stomach_normal | 7 | 1 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9352    0.9712    0.9528       208
     Malignant_lesions     0.9655    0.9545    0.9600       176
body_of_stomach_normal     0.9892    0.9583    0.9735       192

              accuracy                         0.9618       576
             macro avg     0.9633    0.9613    0.9621       576
          weighted avg     0.9625    0.9618    0.9619       576

```