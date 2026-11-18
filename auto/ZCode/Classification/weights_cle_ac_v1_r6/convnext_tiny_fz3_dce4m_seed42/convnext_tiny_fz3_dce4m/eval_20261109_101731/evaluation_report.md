# convnext_tiny_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_fz3_dce4m_seed42/convnext_tiny_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9527 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9530 |
| Balanced Acc | 0.9523 |
| Macro-Sensitivity | 0.9523 |
| Macro-Specificity | 0.9764 |
| Macro-Precision | 0.9534 |
| AUC-PR (Macro) | 0.9923 |
| Cohen's Kappa | 0.9295 |
| MCC | 0.9296 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1499.8 (0.67 ms/张) |
| End-to-End FPS | 7.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9674 | 0.9429 | 0.9474 | 0.9727 | 0.9945 | 0.9904 | 208 |
| Malignant_lesions | 0.9205 | 0.9800 | 0.9529 | 0.9364 | 0.9655 | 0.9951 | 0.9888 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9818 | 0.9643 | 0.9742 | 0.9921 | 0.9986 | 0.9976 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=12 TN=356 FN=10
- **Malignant_lesions**: TP=162 FP=8 TN=392 FN=14
- **body_of_stomach_normal**: TP=189 FP=7 TN=377 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 8 | 2 |
| Malignant_lesions | 9 | 162 | 5 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9429    0.9519    0.9474       208
     Malignant_lesions     0.9529    0.9205    0.9364       176
body_of_stomach_normal     0.9643    0.9844    0.9742       192

              accuracy                         0.9531       576
             macro avg     0.9534    0.9523    0.9527       576
          weighted avg     0.9531    0.9531    0.9530       576

```