# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_seed45/convnext_tiny/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9409 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9935 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9411 |
| Balanced Acc | 0.9379 |
| Macro-Sensitivity | 0.9379 |
| Macro-Specificity | 0.9696 |
| Macro-Precision | 0.9479 |
| AUC-PR (Macro) | 0.9880 |
| Cohen's Kappa | 0.9110 |
| MCC | 0.9131 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 703.3 (1.42 ms/张) |
| End-to-End FPS | 110.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9856 | 0.9239 | 0.8798 | 0.9297 | 0.9913 | 0.9918 | 0.9867 | 208 |
| Malignant_lesions | 0.8750 | 0.9900 | 0.9747 | 0.9222 | 0.9474 | 0.9899 | 0.9794 | 176 |
| body_of_stomach_normal | 0.9531 | 0.9948 | 0.9892 | 0.9708 | 0.9770 | 0.9989 | 0.9979 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=205 FP=28 TN=340 FN=3
- **Malignant_lesions**: TP=154 FP=4 TN=396 FN=22
- **body_of_stomach_normal**: TP=183 FP=2 TN=382 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 205 | 3 | 0 |
| Malignant_lesions | 20 | 154 | 2 |
| body_of_stomach_normal | 8 | 1 | 183 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8798    0.9856    0.9297       208
     Malignant_lesions     0.9747    0.8750    0.9222       176
body_of_stomach_normal     0.9892    0.9531    0.9708       192

              accuracy                         0.9410       576
             macro avg     0.9479    0.9379    0.9409       576
          weighted avg     0.9453    0.9410    0.9411       576

```