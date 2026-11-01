# convnext_cmp_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_cmp_dsd_seed42/convnext_cmp_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9441 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9955 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9446 |
| Balanced Acc | 0.9433 |
| Macro-Sensitivity | 0.9433 |
| Macro-Specificity | 0.9720 |
| Macro-Precision | 0.9453 |
| AUC-PR (Macro) | 0.9917 |
| Cohen's Kappa | 0.9164 |
| MCC | 0.9166 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 28.23M |
| Pure-Inference FPS | 825.2 (1.21 ms/张) |
| End-to-End FPS | 224.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9538 | 0.9213 | 0.9387 | 0.9750 | 0.9947 | 0.9911 | 208 |
| Malignant_lesions | 0.9148 | 0.9675 | 0.9253 | 0.9200 | 0.9627 | 0.9937 | 0.9866 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9948 | 0.9892 | 0.9735 | 0.9795 | 0.9982 | 0.9975 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=17 TN=351 FN=9
- **Malignant_lesions**: TP=161 FP=13 TN=387 FN=15
- **body_of_stomach_normal**: TP=184 FP=2 TN=382 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 14 | 161 | 1 |
| body_of_stomach_normal | 3 | 5 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9213    0.9567    0.9387       208
     Malignant_lesions     0.9253    0.9148    0.9200       176
body_of_stomach_normal     0.9892    0.9583    0.9735       192

              accuracy                         0.9444       576
             macro avg     0.9453    0.9433    0.9441       576
          weighted avg     0.9452    0.9444    0.9446       576

```