# convnext_v2abc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2abc_seed43/convnext_v2abc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9579 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9948 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9582 |
| Balanced Acc | 0.9569 |
| Macro-Sensitivity | 0.9569 |
| Macro-Specificity | 0.9788 |
| Macro-Precision | 0.9597 |
| AUC-PR (Macro) | 0.9906 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9376 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.66G |
| FLOPs | 9.31G |
| Params | 28.67M |
| Pure-Inference FPS | 545.7 (1.83 ms/张) |
| End-to-End FPS | 201.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9620 | 0.9349 | 0.9504 | 0.9806 | 0.9931 | 0.9879 | 208 |
| Malignant_lesions | 0.9148 | 0.9875 | 0.9699 | 0.9415 | 0.9634 | 0.9926 | 0.9864 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9870 | 0.9744 | 0.9819 | 0.9948 | 0.9987 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=14 TN=354 FN=7
- **Malignant_lesions**: TP=161 FP=5 TN=395 FN=15
- **body_of_stomach_normal**: TP=190 FP=5 TN=379 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 12 | 161 | 3 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9349    0.9663    0.9504       208
     Malignant_lesions     0.9699    0.9148    0.9415       176
body_of_stomach_normal     0.9744    0.9896    0.9819       192

              accuracy                         0.9583       576
             macro avg     0.9597    0.9569    0.9579       576
          weighted avg     0.9587    0.9583    0.9582       576

```