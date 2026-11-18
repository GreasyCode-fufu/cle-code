# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4e_seed42/convnext_v4e/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9635 |
| Accuracy | 0.9635 |
| Macro-AUROC (OvR) | 0.9950 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9635 |
| Balanced Acc | 0.9637 |
| Macro-Sensitivity | 0.9637 |
| Macro-Specificity | 0.9818 |
| Macro-Precision | 0.9632 |
| AUC-PR (Macro) | 0.9899 |
| Cohen's Kappa | 0.9452 |
| MCC | 0.9452 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 580.7 (1.72 ms/张) |
| End-to-End FPS | 200.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9755 | 0.9563 | 0.9517 | 0.9703 | 0.9932 | 0.9885 | 208 |
| Malignant_lesions | 0.9545 | 0.9750 | 0.9438 | 0.9492 | 0.9799 | 0.9920 | 0.9817 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9948 | 0.9896 | 0.9896 | 0.9948 | 0.9998 | 0.9996 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=9 TN=359 FN=11
- **Malignant_lesions**: TP=168 FP=10 TN=390 FN=8
- **body_of_stomach_normal**: TP=190 FP=2 TN=382 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 7 | 168 | 1 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9563    0.9471    0.9517       208
     Malignant_lesions     0.9438    0.9545    0.9492       176
body_of_stomach_normal     0.9896    0.9896    0.9896       192

              accuracy                         0.9635       576
             macro avg     0.9632    0.9637    0.9635       576
          weighted avg     0.9636    0.9635    0.9635       576

```