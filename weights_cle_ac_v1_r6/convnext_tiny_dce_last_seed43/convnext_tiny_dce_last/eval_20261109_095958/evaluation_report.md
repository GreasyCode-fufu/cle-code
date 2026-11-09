# convnext_tiny_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_dce_last_seed43/convnext_tiny_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9443 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9938 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9444 |
| Balanced Acc | 0.9437 |
| Macro-Sensitivity | 0.9437 |
| Macro-Specificity | 0.9720 |
| Macro-Precision | 0.9451 |
| AUC-PR (Macro) | 0.9883 |
| Cohen's Kappa | 0.9164 |
| MCC | 0.9165 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1701.0 (0.59 ms/张) |
| End-to-End FPS | 248.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9592 | 0.9292 | 0.9381 | 0.9698 | 0.9920 | 0.9862 | 208 |
| Malignant_lesions | 0.9205 | 0.9775 | 0.9474 | 0.9337 | 0.9654 | 0.9918 | 0.9834 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9792 | 0.9585 | 0.9610 | 0.9817 | 0.9976 | 0.9953 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=15 TN=353 FN=11
- **Malignant_lesions**: TP=162 FP=9 TN=391 FN=14
- **body_of_stomach_normal**: TP=185 FP=8 TN=376 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 8 | 3 |
| Malignant_lesions | 9 | 162 | 5 |
| body_of_stomach_normal | 6 | 1 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9292    0.9471    0.9381       208
     Malignant_lesions     0.9474    0.9205    0.9337       176
body_of_stomach_normal     0.9585    0.9635    0.9610       192

              accuracy                         0.9444       576
             macro avg     0.9451    0.9437    0.9443       576
          weighted avg     0.9446    0.9444    0.9444       576

```