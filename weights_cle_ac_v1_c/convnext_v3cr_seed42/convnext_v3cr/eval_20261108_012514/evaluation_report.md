# convnext_v3cr 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v3cr_seed42/convnext_v3cr/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9541 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9546 |
| Balanced Acc | 0.9533 |
| Macro-Sensitivity | 0.9533 |
| Macro-Specificity | 0.9772 |
| Macro-Precision | 0.9556 |
| AUC-PR (Macro) | 0.9932 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9324 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.27G |
| Params | 28.26M |
| Pure-Inference FPS | 977.2 (1.02 ms/张) |
| End-to-End FPS | 233.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9674 | 0.9437 | 0.9549 | 0.9807 | 0.9962 | 0.9939 | 208 |
| Malignant_lesions | 0.9091 | 0.9850 | 0.9639 | 0.9357 | 0.9610 | 0.9946 | 0.9865 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9792 | 0.9594 | 0.9717 | 0.9921 | 0.9995 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=12 TN=356 FN=7
- **Malignant_lesions**: TP=160 FP=6 TN=394 FN=16
- **body_of_stomach_normal**: TP=189 FP=8 TN=376 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 9 | 160 | 7 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9437    0.9663    0.9549       208
     Malignant_lesions     0.9639    0.9091    0.9357       176
body_of_stomach_normal     0.9594    0.9844    0.9717       192

              accuracy                         0.9549       576
             macro avg     0.9556    0.9533    0.9541       576
          weighted avg     0.9551    0.9549    0.9546       576

```