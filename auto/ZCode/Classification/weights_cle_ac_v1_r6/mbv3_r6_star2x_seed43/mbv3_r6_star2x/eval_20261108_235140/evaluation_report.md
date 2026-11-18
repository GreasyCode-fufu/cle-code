# mbv3_r6_star2x 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star2x_seed43/mbv3_r6_star2x/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9324 |
| Accuracy | 0.9323 |
| Macro-AUROC (OvR) | 0.9935 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9325 |
| Balanced Acc | 0.9303 |
| Macro-Sensitivity | 0.9303 |
| Macro-Specificity | 0.9654 |
| Macro-Precision | 0.9364 |
| AUC-PR (Macro) | 0.9878 |
| Cohen's Kappa | 0.8980 |
| MCC | 0.8989 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.10M |
| FLOPs | 464.20M |
| Params | 3.39M |
| Pure-Inference FPS | 5354.0 (0.19 ms/张) |
| End-to-End FPS | 264.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9266 | 0.8800 | 0.9145 | 0.9715 | 0.9905 | 0.9839 | 208 |
| Malignant_lesions | 0.8807 | 0.9775 | 0.9451 | 0.9118 | 0.9490 | 0.9909 | 0.9812 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9922 | 0.9840 | 0.9710 | 0.9794 | 0.9990 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=27 TN=341 FN=10
- **Malignant_lesions**: TP=155 FP=9 TN=391 FN=21
- **body_of_stomach_normal**: TP=184 FP=3 TN=381 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 8 | 2 |
| Malignant_lesions | 20 | 155 | 1 |
| body_of_stomach_normal | 7 | 1 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8800    0.9519    0.9145       208
     Malignant_lesions     0.9451    0.8807    0.9118       176
body_of_stomach_normal     0.9840    0.9583    0.9710       192

              accuracy                         0.9323       576
             macro avg     0.9364    0.9303    0.9324       576
          weighted avg     0.9346    0.9323    0.9325       576

```