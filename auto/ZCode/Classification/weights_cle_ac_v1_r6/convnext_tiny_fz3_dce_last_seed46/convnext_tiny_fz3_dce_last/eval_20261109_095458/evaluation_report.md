# convnext_tiny_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_fz3_dce_last_seed46/convnext_tiny_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9492 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9950 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9495 |
| Balanced Acc | 0.9486 |
| Macro-Sensitivity | 0.9486 |
| Macro-Specificity | 0.9746 |
| Macro-Precision | 0.9500 |
| AUC-PR (Macro) | 0.9897 |
| Cohen's Kappa | 0.9242 |
| MCC | 0.9244 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1507.8 (0.66 ms/张) |
| End-to-End FPS | 22.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9620 | 0.9340 | 0.9429 | 0.9725 | 0.9938 | 0.9906 | 208 |
| Malignant_lesions | 0.9148 | 0.9775 | 0.9471 | 0.9306 | 0.9631 | 0.9923 | 0.9805 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9844 | 0.9691 | 0.9741 | 0.9895 | 0.9989 | 0.9979 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=14 TN=354 FN=10
- **Malignant_lesions**: TP=161 FP=9 TN=391 FN=15
- **body_of_stomach_normal**: TP=188 FP=6 TN=378 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 9 | 1 |
| Malignant_lesions | 10 | 161 | 5 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9340    0.9519    0.9429       208
     Malignant_lesions     0.9471    0.9148    0.9306       176
body_of_stomach_normal     0.9691    0.9792    0.9741       192

              accuracy                         0.9497       576
             macro avg     0.9500    0.9486    0.9492       576
          weighted avg     0.9497    0.9497    0.9495       576

```